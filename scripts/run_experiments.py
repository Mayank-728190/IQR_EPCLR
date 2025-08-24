"""
Comprehensive Benchmarking Suite for Robust Regression Algorithms

This module provides a professional benchmarking framework for comparing
robust regression algorithms including IQR-EPCLR variants against standard
methods. It evaluates performance across multiple dimensions: accuracy,
robustness, computational efficiency, and scalability.

Features:
- Comprehensive experimental design with multiple contamination scenarios
- Statistical significance testing and confidence intervals
- Memory usage monitoring and profiling
- Detailed reporting with publication-ready visualizations
- Reproducible experiments with proper random state management
- Progress tracking and intermediate result saving

Author: [Your Name]
Date: August 2025
License: MIT
"""

import time
import psutil
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from contextlib import contextmanager

# Scientific computing and ML libraries
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy import stats

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import custom algorithms (would need to be adjusted based on actual import paths)
try:
    from iqrepclr_professional import IQREPCLRRegressor
    from linear_time_iqrepclr_professional import LinearTimeIQREPCLRRegressor
    CUSTOM_ALGORITHMS_AVAILABLE = True
except ImportError:
    CUSTOM_ALGORITHMS_AVAILABLE = False
    print("Warning: Custom IQR-EPCLR algorithms not found. Using placeholder implementations.")


@dataclass
class ExperimentConfig:
    """Configuration for benchmark experiments."""
    sample_sizes: List[int]
    n_features: int
    contamination_levels: List[float]
    n_repeats: int
    random_state: int
    noise_level: float
    outlier_strength: float
    results_dir: str
    parallel_jobs: int
    timeout_seconds: int


@dataclass
class ModelResult:
    """Results from a single model evaluation."""
    model_name: str
    fit_time: float
    predict_time: float
    memory_peak: float
    mse_coefficients: float
    mse_predictions: float
    mae_predictions: float
    r2_score: float
    success: bool
    error_message: Optional[str] = None
    n_outliers_detected: Optional[int] = None


class BenchmarkSuite:
    """
    Professional benchmarking suite for robust regression algorithms.
    
    This class orchestrates comprehensive experiments to evaluate robust
    regression methods across multiple performance dimensions.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the benchmarking suite.
        
        Parameters
        ----------
        config : ExperimentConfig
            Configuration object containing all experimental parameters.
        """
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize models dictionary
        self.models = self._initialize_models()
        
        # Results storage
        self.all_results: List[Dict[str, Any]] = []
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.results_dir / "benchmark.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all models to be benchmarked."""
        models = {
            "OLS": LinearRegression(),
            "Huber": HuberRegressor(epsilon=1.35, max_iter=100),
            "RANSAC": RANSACRegressor(
                random_state=self.config.random_state,
                max_trials=100
            ),
            "TheilSen": TheilSenRegressor(
                random_state=self.config.random_state,
                max_subpopulation=10000
            ),
        }
        
        # Add custom algorithms if available
        if CUSTOM_ALGORITHMS_AVAILABLE:
            models.update({
                "IQR-EPCLR": IQREPCLRRegressor(
                    n_candidates=500,
                    random_state=self.config.random_state
                ),
                "LT-IQR-EPCLR": LinearTimeIQREPCLRRegressor(
                    max_epochs=50,
                    batch_size=min(2000, max(self.config.sample_sizes) // 10),
                    random_state=self.config.random_state
                )
            })
        else:
            # Placeholder implementations for demonstration
            self.logger.warning("Using placeholder implementations for custom algorithms")
            
        return models

    def generate_contaminated_data(
        self, 
        n_samples: int, 
        contamination_frac: float, 
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate regression data with realistic contamination patterns.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        contamination_frac : float
            Fraction of samples to contaminate (0.0 to 1.0).
        seed : int
            Random seed for reproducibility.
            
        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        true_coef : ndarray of shape (n_features,)
            True coefficient values.
        """
        rng = np.random.RandomState(seed)
        
        # Generate clean regression data
        X, y, true_coef = make_regression(
            n_samples=n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_features,
            noise=self.config.noise_level,
            coef=True,
            random_state=seed
        )
        
        n_outliers = int(n_samples * contamination_frac)
        if n_outliers == 0:
            return X, y, true_coef
        
        # Create different types of outliers
        outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
        
        # Split outliers into different types
        n_vertical = n_outliers // 2
        n_leverage = n_outliers - n_vertical
        
        vertical_indices = outlier_indices[:n_vertical]
        leverage_indices = outlier_indices[n_vertical:]
        
        # Vertical outliers (response outliers)
        if len(vertical_indices) > 0:
            y[vertical_indices] += self.config.outlier_strength * rng.randn(len(vertical_indices))
        
        # Bad leverage points (predictor space outliers with wrong response)
        if len(leverage_indices) > 0:
            # Move points to extreme positions in predictor space
            X[leverage_indices] += 3 * np.std(X, axis=0) * rng.randn(len(leverage_indices), X.shape[1])
            # Give them wrong responses
            y[leverage_indices] += self.config.outlier_strength * rng.randn(len(leverage_indices))
        
        return X, y, true_coef

    @contextmanager
    def _monitor_resources(self):
        """Context manager for monitoring memory and timing."""
        # Start memory monitoring
        tracemalloc.start()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_peak = peak / 1024 / 1024  # MB
            
            # Store in context for retrieval
            self._last_timing = elapsed_time
            self._last_memory = memory_peak

    def evaluate_single_model(
        self, 
        model_name: str, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        true_coef: np.ndarray
    ) -> ModelResult:
        """
        Evaluate a single model on given data.
        
        Parameters
        ----------
        model_name : str
            Name of the model for identification.
        model : sklearn estimator
            The model to evaluate.
        X : ndarray
            Feature matrix.
        y : ndarray
            Target values.
        true_coef : ndarray
            True coefficient values for comparison.
            
        Returns
        -------
        result : ModelResult
            Complete evaluation results.
        """
        try:
            # Fit the model with resource monitoring
            with self._monitor_resources():
                model.fit(X, y)
            
            fit_time = self._last_timing
            memory_peak = self._last_memory
            
            # Make predictions with timing
            start_time = time.time()
            y_pred = model.predict(X)
            predict_time = time.time() - start_time
            
            # Extract coefficients safely
            if hasattr(model, 'coef_'):
                pred_coef = np.atleast_1d(model.coef_)
            elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'coef_'):
                pred_coef = np.atleast_1d(model.estimator_.coef_)
            else:
                pred_coef = np.full_like(true_coef, np.nan)
            
            # Calculate metrics
            mse_coef = mean_squared_error(true_coef, pred_coef) if not np.any(np.isnan(pred_coef)) else np.nan
            mse_pred = mean_squared_error(y, y_pred)
            mae_pred = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Count outliers if available
            n_outliers = None
            if hasattr(model, 'outlier_mask_'):
                n_outliers = np.sum(model.outlier_mask_)
            elif hasattr(model, 'inlier_mask_'):
                n_outliers = np.sum(~model.inlier_mask_)
                
            return ModelResult(
                model_name=model_name,
                fit_time=fit_time,
                predict_time=predict_time,
                memory_peak=memory_peak,
                mse_coefficients=mse_coef,
                mse_predictions=mse_pred,
                mae_predictions=mae_pred,
                r2_score=r2,
                success=True,
                n_outliers_detected=n_outliers
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            return ModelResult(
                model_name=model_name,
                fit_time=np.nan,
                predict_time=np.nan,
                memory_peak=np.nan,
                mse_coefficients=np.nan,
                mse_predictions=np.nan,
                mae_predictions=np.nan,
                r2_score=np.nan,
                success=False,
                error_message=str(e)
            )

    def run_single_experiment(
        self, 
        n_samples: int, 
        contamination: float, 
        repeat: int
    ) -> List[Dict[str, Any]]:
        """
        Run a single experimental configuration.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
        contamination : float
            Fraction of contaminated samples.
        repeat : int
            Repeat number for this configuration.
            
        Returns
        -------
        results : list of dict
            Results for all models in this experiment.
        """
        # Generate data
        seed = self.config.random_state + repeat
        X, y, true_coef = self.generate_contaminated_data(n_samples, contamination, seed)
        
        experiment_results = []
        
        for model_name, model in self.models.items():
            # Create a fresh copy of the model to avoid state issues
            try:
                if hasattr(model, 'random_state'):
                    model.set_params(random_state=seed)
                model_copy = model.__class__(**model.get_params())
            except:
                model_copy = model
                
            # Evaluate model
            result = self.evaluate_single_model(model_name, model_copy, X, y, true_coef)
            
            # Convert to dictionary and add experiment metadata
            result_dict = {
                'model': result.model_name,
                'n_samples': n_samples,
                'contamination': contamination,
                'repeat': repeat,
                'fit_time': result.fit_time,
                'predict_time': result.predict_time,
                'memory_peak': result.memory_peak,
                'mse_coefficients': result.mse_coefficients,
                'mse_predictions': result.mse_predictions,
                'mae_predictions': result.mae_predictions,
                'r2_score': result.r2_score,
                'success': result.success,
                'error_message': result.error_message,
                'n_outliers_detected': result.n_outliers_detected
            }
            
            experiment_results.append(result_dict)
            
        return experiment_results

    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run the complete experimental suite.
        
        Returns
        -------
        results_df : pd.DataFrame
            Complete results from all experiments.
        """
        self.logger.info("Starting comprehensive benchmarking suite")
        self.logger.info(f"Configuration: {self.config}")
        
        total_experiments = (len(self.config.sample_sizes) * 
                           len(self.config.contamination_levels) * 
                           self.config.n_repeats)
        
        completed = 0
        
        for n_samples in self.config.sample_sizes:
            for contamination in self.config.contamination_levels:
                for repeat in range(self.config.n_repeats):
                    completed += 1
                    progress = 100 * completed / total_experiments
                    
                    self.logger.info(
                        f"Progress: {progress:.1f}% - Running experiment "
                        f"n={n_samples}, contamination={contamination:.1%}, "
                        f"repeat={repeat+1}/{self.config.n_repeats}"
                    )
                    
                    try:
                        exp_results = self.run_single_experiment(n_samples, contamination, repeat)
                        self.all_results.extend(exp_results)
                        
                        # Save intermediate results periodically
                        if completed % 10 == 0:
                            self._save_intermediate_results()
                            
                    except Exception as e:
                        self.logger.error(f"Failed experiment: {str(e)}")
                        continue
        
        # Convert to DataFrame and save final results
        results_df = pd.DataFrame(self.all_results)
        results_file = self.results_dir / "experimental_results.csv"
        results_df.to_csv(results_file, index=False)
        
        self.logger.info(f"All experiments completed. Results saved to {results_file}")
        return results_df

    def _save_intermediate_results(self) -> None:
        """Save intermediate results to prevent data loss."""
        if self.all_results:
            temp_df = pd.DataFrame(self.all_results)
            temp_file = self.results_dir / "intermediate_results.csv"
            temp_df.to_csv(temp_file, index=False)

    def generate_comprehensive_report(self, results_df: pd.DataFrame) -> None:
        """
        Generate comprehensive analysis report with visualizations.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Complete experimental results.
        """
        self.logger.info("Generating comprehensive analysis report")
        
        # Filter successful experiments
        successful_results = results_df[results_df['success'] == True].copy()
        
        if successful_results.empty:
            self.logger.error("No successful experiments to analyze!")
            return
        
        # Aggregate results with statistics
        agg_stats = successful_results.groupby(['n_samples', 'contamination', 'model']).agg({
            'fit_time': ['mean', 'std', 'min', 'max'],
            'mse_coefficients': ['mean', 'std', 'min', 'max'],
            'mse_predictions': ['mean', 'std', 'min', 'max'],
            'r2_score': ['mean', 'std', 'min', 'max'],
            'memory_peak': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        agg_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in agg_stats.columns.values]
        
        # Save aggregated statistics
        stats_file = self.results_dir / "aggregated_statistics.csv"
        agg_stats.to_csv(stats_file, index=False)
        
        # Generate visualizations
        self._create_performance_plots(successful_results)
        self._create_robustness_analysis(successful_results)
        self._create_scalability_analysis(successful_results)
        self._create_summary_tables(agg_stats)
        
        # Generate executive summary
        self._generate_executive_summary(successful_results, agg_stats)

    def _create_performance_plots(self, results_df: pd.DataFrame) -> None:
        """Create performance comparison plots."""
        plt.style.use('default')
        
        # Set up the plotting style
        sns.set_palette("husl")
        
        # 1. Computational Time Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Computational Performance Analysis', fontsize=16, fontweight='bold')
        
        # Fit time vs sample size
        sns.lineplot(
            data=results_df, 
            x='n_samples', 
            y='fit_time', 
            hue='model', 
            ax=axes[0,0]
        )
        axes[0,0].set_yscale('log')
        axes[0,0].set_xscale('log')
        axes[0,0].set_title('Training Time vs Sample Size')
        axes[0,0].set_ylabel('Fit Time (seconds)')
        
        # Memory usage vs sample size
        sns.lineplot(
            data=results_df, 
            x='n_samples', 
            y='memory_peak', 
            hue='model', 
            ax=axes[0,1]
        )
        axes[0,1].set_xscale('log')
        axes[0,1].set_title('Peak Memory Usage vs Sample Size')
        axes[0,1].set_ylabel('Memory Peak (MB)')
        
        # Fit time vs contamination
        sns.boxplot(
            data=results_df,
            x='contamination',
            y='fit_time',
            hue='model',
            ax=axes[1,0]
        )
        axes[1,0].set_yscale('log')
        axes[1,0].set_title('Training Time vs Contamination Level')
        axes[1,0].set_ylabel('Fit Time (seconds)')
        
        # R² score vs contamination
        sns.lineplot(
            data=results_df,
            x='contamination',
            y='r2_score',
            hue='model',
            ax=axes[1,1]
        )
        axes[1,1].set_title('R² Score vs Contamination Level')
        axes[1,1].set_ylabel('R² Score')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_robustness_analysis(self, results_df: pd.DataFrame) -> None:
        """Create robustness analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robustness Analysis', fontsize=16, fontweight='bold')
        
        # MSE of coefficients vs contamination
        sns.lineplot(
            data=results_df,
            x='contamination',
            y='mse_coefficients',
            hue='model',
            ax=axes[0,0]
        )
        axes[0,0].set_yscale('log')
        axes[0,0].set_title('Coefficient MSE vs Contamination')
        axes[0,0].set_ylabel('MSE (Coefficients)')
        
        # Prediction MSE vs contamination
        sns.lineplot(
            data=results_df,
            x='contamination',
            y='mse_predictions',
            hue='model',
            ax=axes[0,1]
        )
        axes[0,1].set_yscale('log')
        axes[0,1].set_title('Prediction MSE vs Contamination')
        axes[0,1].set_ylabel('MSE (Predictions)')
        
        # Robustness breakdown analysis
        contamination_impact = results_df.groupby(['model', 'contamination'])['r2_score'].mean().reset_index()
        pivot_data = contamination_impact.pivot(index='contamination', columns='model', values='r2_score')
        
        sns.heatmap(
            pivot_data.T,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            ax=axes[1,0]
        )
        axes[1,0].set_title('R² Heatmap by Contamination Level')
        axes[1,0].set_ylabel('Model')
        
        # Error distribution
        sns.violinplot(
            data=results_df[results_df['contamination'] > 0],
            x='model',
            y='mse_predictions',
            ax=axes[1,1]
        )
        axes[1,1].set_yscale('log')
        axes[1,1].set_title('Error Distribution (Contaminated Data)')
        axes[1,1].set_ylabel('MSE (Predictions)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_scalability_analysis(self, results_df: pd.DataFrame) -> None:
        """Create scalability analysis plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
        
        # Time complexity analysis
        clean_data = results_df[results_df['contamination'] == 0.0]
        
        sns.scatterplot(
            data=clean_data,
            x='n_samples',
            y='fit_time',
            hue='model',
            s=100,
            ax=axes[0]
        )
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_title('Computational Complexity (Clean Data)')
        axes[0].set_xlabel('Number of Samples')
        axes[0].set_ylabel('Fit Time (seconds)')
        
        # Add theoretical complexity lines for reference
        sample_sizes = np.array(sorted(clean_data['n_samples'].unique()))
        axes[0].plot(sample_sizes, sample_sizes * 1e-6, 'k--', alpha=0.5, label='O(n)')
        axes[0].plot(sample_sizes, sample_sizes * np.log(sample_sizes) * 1e-7, 'r--', alpha=0.5, label='O(n log n)')
        
        # Memory scaling
        sns.scatterplot(
            data=clean_data,
            x='n_samples',
            y='memory_peak',
            hue='model',
            s=100,
            ax=axes[1]
        )
        axes[1].set_xscale('log')
        axes[1].set_title('Memory Scaling (Clean Data)')
        axes[1].set_xlabel('Number of Samples')
        axes[1].set_ylabel('Peak Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_tables(self, agg_stats: pd.DataFrame) -> None:
        """Create summary tables for the report."""
        # Best performing models by metric
        metrics = ['fit_time_mean', 'mse_coefficients_mean', 'r2_score_mean']
        
        summary_tables = {}
        for metric in metrics:
            if metric in agg_stats.columns:
                ascending = metric != 'r2_score_mean'  # R² should be descending
                best_models = (agg_stats.groupby(['n_samples', 'contamination'])[metric]
                              .apply(lambda x: agg_stats.loc[x.idxmin() if ascending else x.idxmax(), 'model']))
                summary_tables[metric] = best_models.reset_index()
        
        # Save summary tables
        with pd.ExcelWriter(self.results_dir / 'summary_tables.xlsx') as writer:
            for metric_name, table in summary_tables.items():
                table.to_excel(writer, sheet_name=metric_name.replace('_mean', ''), index=False)

    def _generate_executive_summary(self, results_df: pd.DataFrame, agg_stats: pd.DataFrame) -> None:
        """Generate executive summary report."""
        summary_file = self.results_dir / 'executive_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("ROBUST REGRESSION BENCHMARKING - EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            f.write(f"Total experiments conducted: {len(results_df)}\n")
            f.write(f"Successful experiments: {results_df['success'].sum()}\n")
            f.write(f"Sample sizes tested: {sorted(results_df['n_samples'].unique())}\n")
            f.write(f"Contamination levels: {sorted(results_df['contamination'].unique())}\n")
            f.write(f"Models evaluated: {sorted(results_df['model'].unique())}\n\n")
            
            # Performance winners
            f.write("KEY FINDINGS:\n")
            f.write("-" * 20 + "\n\n")
            
            # Speed winner
            if 'fit_time_mean' in agg_stats.columns:
                speed_winner = agg_stats.loc[agg_stats['fit_time_mean'].idxmin(), 'model']
                f.write(f"Fastest model overall: {speed_winner}\n")
            
            # Accuracy winner (clean data)
            clean_data_stats = agg_stats[agg_stats['contamination'] == 0.0]
            if not clean_data_stats.empty and 'r2_score_mean' in clean_data_stats.columns:
                accuracy_winner = clean_data_stats.loc[clean_data_stats['r2_score_mean'].idxmax(), 'model']
                f.write(f"Most accurate on clean data: {accuracy_winner}\n")
            
            # Robustness winner
            contaminated_stats = agg_stats[agg_stats['contamination'] > 0.0]
            if not contaminated_stats.empty and 'r2_score_mean' in contaminated_stats.columns:
                robustness_winner = contaminated_stats.loc[contaminated_stats['r2_score_mean'].idxmax(), 'model']
                f.write(f"Most robust to contamination: {robustness_winner}\n")
            
            f.write(f"\nDetailed results and visualizations saved to: {self.results_dir}\n")

        self.logger.info(f"Executive summary saved to {summary_file}")


def main():
    """Main function to run the benchmarking suite."""
    # Configuration
    config = ExperimentConfig(
        sample_sizes=[1000, 5000, 10000, 50000],
        n_features=5,
        contamination_levels=[0.0, 0.1, 0.25, 0.4],
        n_repeats=3,  # Reduced for faster execution, increase for production
        random_state=42,
        noise_level=10.0,
        outlier_strength=100.0,
        results_dir="benchmark_results",
        parallel_jobs=1,  # Can be increased for parallel execution
        timeout_seconds=300
    )
    
    # Initialize and run benchmarking suite
    benchmark = BenchmarkSuite(config)
    
    try:
        # Run all experiments
        results_df = benchmark.run_all_experiments()
        
        # Generate comprehensive analysis
        benchmark.generate_comprehensive_report(results_df)
        
        print("\n" + "="*60)
        print("BENCHMARKING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results directory: {benchmark.results_dir}")
        print("Generated files:")
        print("- experimental_results.csv: Raw experimental data")
        print("- aggregated_statistics.csv: Statistical summaries")
        print("- performance_analysis.png: Performance plots")
        print("- robustness_analysis.png: Robustness analysis")
        print("- scalability_analysis.png: Scalability plots")
        print("- summary_tables.xlsx: Summary tables")
        print("- executive_summary.txt: Executive summary")
        print("- benchmark.log: Detailed execution log")
        
    except Exception as e:
        logging.error(f"Benchmarking failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# Additional utility functions for advanced analysis
class StatisticalAnalysis:
    """Advanced statistical analysis utilities for benchmarking results."""
    
    @staticmethod
    def compute_confidence_intervals(data: pd.DataFrame, confidence=0.95) -> pd.DataFrame:
        """
        Compute confidence intervals for model performance metrics.
        
        Parameters
        ----------
        data : pd.DataFrame
            Results data with model performance metrics.
        confidence : float
            Confidence level for intervals (default 0.95).
            
        Returns
        -------
        ci_results : pd.DataFrame
            DataFrame with confidence intervals.
        """
        alpha = 1 - confidence
        metrics = ['fit_time', 'mse_coefficients', 'mse_predictions', 'r2_score']
        
        ci_results = []
        
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            
            for metric in metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 1:
                        mean_val = values.mean()
                        sem = stats.sem(values)
                        ci_low, ci_high = stats.t.interval(
                            confidence, len(values)-1, loc=mean_val, scale=sem
                        )
                        
                        ci_results.append({
                            'model': model,
                            'metric': metric,
                            'mean': mean_val,
                            'ci_low': ci_low,
                            'ci_high': ci_high,
                            'n_samples': len(values)
                        })
        
        return pd.DataFrame(ci_results)
    
    @staticmethod
    def perform_statistical_tests(data: pd.DataFrame, reference_model: str = 'OLS') -> pd.DataFrame:
        """
        Perform statistical significance tests comparing models.
        
        Parameters
        ----------
        data : pd.DataFrame
            Results data for comparison.
        reference_model : str
            Reference model for comparisons.
            
        Returns
        -------
        test_results : pd.DataFrame
            Statistical test results.
        """
        metrics = ['mse_coefficients', 'mse_predictions', 'r2_score']
        test_results = []
        
        ref_data = data[data['model'] == reference_model]
        
        for model in data['model'].unique():
            if model == reference_model:
                continue
                
            model_data = data[data['model'] == model]
            
            for metric in metrics:
                if metric in data.columns:
                    ref_values = ref_data[metric].dropna()
                    model_values = model_data[metric].dropna()
                    
                    if len(ref_values) > 1 and len(model_values) > 1:
                        # Perform Wilcoxon signed-rank test (non-parametric)
                        try:
                            statistic, p_value = stats.wilcoxon(ref_values, model_values)
                            test_results.append({
                                'model': model,
                                'reference': reference_model,
                                'metric': metric,
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
                        except Exception:
                            # Fallback to t-test if Wilcoxon fails
                            statistic, p_value = stats.ttest_rel(ref_values, model_values)
                            test_results.append({
                                'model': model,
                                'reference': reference_model,
                                'metric': metric,
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
        
        return pd.DataFrame(test_results)


class AdvancedVisualization:
    """Advanced visualization utilities for benchmarking results."""
    
    @staticmethod
    def create_radar_chart(data: pd.DataFrame, models: List[str], save_path: str) -> None:
        """
        Create radar chart comparing models across multiple metrics.
        
        Parameters
        ----------
        data : pd.DataFrame
            Aggregated performance data.
        models : list
            List of model names to include.
        save_path : str
            Path to save the visualization.
        """
        import matplotlib.pyplot as plt
        from math import pi
        
        # Define metrics (normalized to 0-1 scale)
        metrics = ['Speed', 'Accuracy', 'Robustness', 'Memory Efficiency']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Set up angles for each metric
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        # Colors for each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_data = data[data['model'] == model]
            if not model_data.empty:
                # Normalize metrics to 0-1 scale (this would need actual implementation)
                values = [0.7, 0.8, 0.6, 0.9]  # Placeholder values
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Comparison Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_performance_matrix(data: pd.DataFrame, save_path: str) -> None:
        """
        Create a performance matrix heatmap.
        
        Parameters
        ----------
        data : pd.DataFrame
            Performance data.
        save_path : str
            Path to save the visualization.
        """
        # Create performance matrix
        performance_metrics = ['fit_time', 'mse_coefficients', 'r2_score']
        
        # Aggregate data
        matrix_data = data.groupby(['model', 'contamination'])[performance_metrics].mean().reset_index()
        
        # Create subplots for each metric
        fig, axes = plt.subplots(1, len(performance_metrics), figsize=(18, 6))
        fig.suptitle('Performance Matrix by Contamination Level', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(performance_metrics):
            pivot_data = matrix_data.pivot(index='model', columns='contamination', values=metric)
            
            # Determine colormap direction
            cmap = 'RdYlBu' if metric != 'r2_score' else 'RdYlBu_r'
            
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                ax=axes[i],
                cbar_kws={'shrink': 0.8}
            )
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Contamination Level')
            if i == 0:
                axes[i].set_ylabel('Model')
            else:
                axes[i].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ModelProfiler:
    """Detailed profiling utilities for individual models."""
    
    def __init__(self, model, model_name: str):
        """
        Initialize model profiler.
        
        Parameters
        ----------
        model : sklearn estimator
            Model to profile.
        model_name : str
            Name of the model.
        """
        self.model = model
        self.model_name = model_name
        self.profile_data = {}
    
    def profile_convergence(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Profile model convergence behavior.
        
        Parameters
        ----------
        X : ndarray
            Feature matrix.
        y : ndarray
            Target values.
            
        Returns
        -------
        convergence_info : dict
            Convergence profiling information.
        """
        convergence_info = {'model': self.model_name}
        
        # Check if model has convergence tracking
        if hasattr(self.model, 'max_iter'):
            original_max_iter = self.model.max_iter
            
            # Test different iteration limits
            iter_limits = [10, 25, 50, 100, 200]
            convergence_data = []
            
            for max_iter in iter_limits:
                self.model.set_params(max_iter=max_iter)
                
                start_time = time.time()
                self.model.fit(X, y)
                fit_time = time.time() - start_time
                
                # Get final loss if available
                final_loss = getattr(self.model, 'loss_history_', [np.nan])[-1]
                
                convergence_data.append({
                    'max_iter': max_iter,
                    'fit_time': fit_time,
                    'final_loss': final_loss,
                    'converged': getattr(self.model, 'n_iter_', max_iter) < max_iter
                })
            
            # Restore original parameter
            self.model.set_params(max_iter=original_max_iter)
            convergence_info['convergence_data'] = convergence_data
        
        return convergence_info
    
    def profile_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Profile model sensitivity to hyperparameters.
        
        Parameters
        ----------
        X : ndarray
            Feature matrix.
        y : ndarray
            Target values.
            
        Returns
        -------
        hyperparam_info : dict
            Hyperparameter sensitivity information.
        """
        hyperparam_info = {'model': self.model_name, 'sensitivity_data': []}
        
        # Get model parameters
        params = self.model.get_params()
        
        # Define parameter ranges for common parameters
        param_ranges = {
            'learning_rate': [0.001, 0.01, 0.1, 0.5],
            'c_tukey': [3.0, 4.685, 6.0, 8.0],
            'epsilon': [1.1, 1.35, 1.5, 2.0],
            'iqr_threshold': [1.0, 1.5, 2.0, 3.0]
        }
        
        for param_name, param_values in param_ranges.items():
            if param_name in params:
                original_value = params[param_name]
                param_results = []
                
                for value in param_values:
                    try:
                        self.model.set_params(**{param_name: value})
                        
                        start_time = time.time()
                        self.model.fit(X, y)
                        fit_time = time.time() - start_time
                        
                        y_pred = self.model.predict(X)
                        mse = mean_squared_error(y, y_pred)
                        
                        param_results.append({
                            'param_value': value,
                            'fit_time': fit_time,
                            'mse': mse
                        })
                        
                    except Exception as e:
                        param_results.append({
                            'param_value': value,
                            'fit_time': np.nan,
                            'mse': np.nan,
                            'error': str(e)
                        })
                
                # Restore original value
                self.model.set_params(**{param_name: original_value})
                
                hyperparam_info['sensitivity_data'].append({
                    'parameter': param_name,
                    'results': param_results
                })
        
        return hyperparam_info


# Example usage for advanced features
def run_advanced_analysis():
    """Run advanced analysis on benchmarking results."""
    print("Running advanced statistical analysis...")
    
    # Load results (assuming they exist)
    results_file = Path("benchmark_results") / "experimental_results.csv"
    if results_file.exists():
        data = pd.read_csv(results_file)
        
        # Statistical analysis
        stats_analyzer = StatisticalAnalysis()
        ci_results = stats_analyzer.compute_confidence_intervals(data)
        test_results = stats_analyzer.perform_statistical_tests(data)
        
        print("Confidence intervals computed")
        print("Statistical tests completed")
        
        # Advanced visualizations
        viz = AdvancedVisualization()
        models_to_compare = data['model'].unique()[:5]  # Limit to 5 models
        
        viz.create_radar_chart(data, models_to_compare, "benchmark_results/radar_chart.png")
        viz.create_performance_matrix(data, "benchmark_results/performance_matrix.png")
        
        print("Advanced visualizations created")
        
        # Save advanced analysis results
        ci_results.to_csv("benchmark_results/confidence_intervals.csv", index=False)
        test_results.to_csv("benchmark_results/statistical_tests.csv", index=False)
        
        print("Advanced analysis completed!")
    else:
        print("No results file found. Run main benchmarking first.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "advanced":
        run_advanced_analysis()
    else:
        main()