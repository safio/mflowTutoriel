"""
MLflow Model Registry Manager

This module provides comprehensive utilities for MLflow Model Registry operations including:
- Model registration and versioning
- Model lifecycle management (staging: None → Staging → Production → Archived)
- Model comparison and rollback capabilities
- Automated model promotion based on metrics
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLflowRegistryManager:
    """Comprehensive manager for MLflow Model Registry operations"""

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow Registry Manager

        Args:
            tracking_uri: MLflow tracking server URI (defaults to local ./mlruns)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()
        logger.info(f"✅ MLflow Registry Manager initialized (Tracking URI: {mlflow.get_tracking_uri()})")

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model from an MLflow run to the Model Registry

        Args:
            run_id: MLflow run ID containing the model
            model_name: Name to register the model under
            artifact_path: Path to model artifact in the run (default: "model")
            tags: Optional tags for the model version
            description: Optional description for the model version

        Returns:
            Model version number as string
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"

            logger.info(f"📦 Registering model '{model_name}' from run {run_id[:8]}...")
            model_version = mlflow.register_model(model_uri, model_name)

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=str(value)
                    )
                logger.info(f"   🏷️  Added {len(tags)} tags")

            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
                logger.info(f"   📝 Updated description")

            logger.info(f"✅ Model registered: {model_name} v{model_version.version}")
            return str(model_version.version)

        except Exception as e:
            logger.error(f"❌ Error registering model: {str(e)}")
            raise

    def create_registered_model(
        self,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Create a new registered model in the Model Registry

        Args:
            model_name: Name of the model to create
            tags: Optional tags for the registered model
            description: Optional description
        """
        try:
            self.client.create_registered_model(
                name=model_name,
                tags=tags,
                description=description
            )
            logger.info(f"✅ Created registered model: '{model_name}'")
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                logger.warning(f"⚠️  Model '{model_name}' already exists")
            else:
                logger.error(f"❌ Error creating registered model: {str(e)}")
                raise

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """
        Transition a model version to a new stage

        Note: Model stages are deprecated in MLflow 2.9.0+. Consider using aliases instead.
        This method is kept for backward compatibility.

        Args:
            model_name: Name of the registered model
            version: Model version number
            stage: Target stage ('None', 'Staging', 'Production', 'Archived')
            archive_existing_versions: Whether to archive existing versions in target stage
        """
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage,
                    archive_existing_versions=archive_existing_versions
                )
            logger.info(f"🔄 Transitioned '{model_name}' v{version} to '{stage}' stage")

            if archive_existing_versions:
                logger.info(f"   📦 Archived existing versions in '{stage}' stage")

        except Exception as e:
            logger.error(f"❌ Error transitioning model stage: {str(e)}")
            # Try using aliases as fallback for newer MLflow versions
            try:
                logger.info(f"   Attempting to use alias instead of stage...")
                self.set_model_alias(model_name, version, stage.lower())
            except Exception as alias_error:
                logger.error(f"❌ Alias fallback also failed: {str(alias_error)}")
                raise e

    def set_model_alias(
        self,
        model_name: str,
        version: str,
        alias: str
    ) -> None:
        """
        Set an alias for a model version (MLflow 2.9.0+ recommended approach)

        Args:
            model_name: Name of the registered model
            version: Model version number
            alias: Alias name (e.g., 'production', 'staging', 'champion')
        """
        try:
            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version
            )
            logger.info(f"✅ Set alias '{alias}' for '{model_name}' v{version}")
        except AttributeError:
            # Older MLflow version doesn't support aliases
            logger.warning(f"⚠️  Model aliases not supported in this MLflow version")
            raise
        except Exception as e:
            logger.error(f"❌ Error setting model alias: {str(e)}")
            raise

    def promote_model(
        self,
        model_name: str,
        version: str,
        from_stage: str = "Staging",
        to_stage: str = "Production",
        archive_existing: bool = True
    ) -> None:
        """
        Promote a model from one stage to another

        Args:
            model_name: Name of the registered model
            version: Model version to promote
            from_stage: Current stage (for validation)
            to_stage: Target stage
            archive_existing: Whether to archive existing models in target stage
        """
        # Verify current stage
        model_version = self.client.get_model_version(model_name, version)
        current_stage = model_version.current_stage

        if current_stage != from_stage:
            logger.warning(
                f"⚠️  Model is in '{current_stage}' stage, not '{from_stage}'. "
                f"Proceeding with promotion anyway."
            )

        # Promote
        self.transition_model_stage(
            model_name=model_name,
            version=version,
            stage=to_stage,
            archive_existing_versions=archive_existing
        )

        logger.info(f"🚀 Promoted '{model_name}' v{version}: {from_stage} → {to_stage}")

    def rollback_model(
        self,
        model_name: str,
        target_version: str,
        stage: str = "Production"
    ) -> None:
        """
        Rollback to a previous model version

        Args:
            model_name: Name of the registered model
            target_version: Version to rollback to
            stage: Stage to set for the target version (default: 'Production')
        """
        # Archive current models in the target stage
        current_versions = self.get_model_versions_by_stage(model_name, stage)

        for version_info in current_versions:
            if version_info['version'] != target_version:
                self.transition_model_stage(
                    model_name=model_name,
                    version=version_info['version'],
                    stage="Archived"
                )

        # Promote target version
        self.transition_model_stage(
            model_name=model_name,
            version=target_version,
            stage=stage
        )

        logger.info(f"⏪ Rolled back '{model_name}' to version {target_version} in '{stage}'")

    def get_latest_model_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model, optionally filtered by stage

        Args:
            model_name: Name of the registered model
            stage: Optional stage filter ('Staging', 'Production', etc.)

        Returns:
            Dictionary with model version information or None if not found
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                all_versions = self.client.search_model_versions(f"name='{model_name}'")
                if not all_versions:
                    return None
                versions = [max(all_versions, key=lambda x: int(x.version))]

            if not versions:
                return None

            version = versions[0]
            return {
                'name': version.name,
                'version': version.version,
                'stage': version.current_stage,
                'run_id': version.run_id,
                'status': version.status,
                'creation_timestamp': version.creation_timestamp,
                'last_updated_timestamp': version.last_updated_timestamp
            }

        except Exception as e:
            logger.error(f"❌ Error getting latest model version: {str(e)}")
            return None

    def get_model_versions_by_stage(
        self,
        model_name: str,
        stage: str
    ) -> List[Dict[str, Any]]:
        """
        Get all versions of a model in a specific stage

        Args:
            model_name: Name of the registered model
            stage: Stage to filter by

        Returns:
            List of model version dictionaries
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            return [
                {
                    'name': v.name,
                    'version': v.version,
                    'stage': v.current_stage,
                    'run_id': v.run_id,
                    'status': v.status
                }
                for v in versions
            ]
        except Exception as e:
            logger.error(f"❌ Error getting model versions: {str(e)}")
            return []

    def compare_model_versions(
        self,
        model_name: str,
        version1: str,
        version2: str,
        metric_keys: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics between two model versions

        Args:
            model_name: Name of the registered model
            version1: First version to compare
            version2: Second version to compare
            metric_keys: Optional list of specific metrics to compare

        Returns:
            DataFrame with comparison results
        """
        try:
            # Get model versions
            mv1 = self.client.get_model_version(model_name, version1)
            mv2 = self.client.get_model_version(model_name, version2)

            # Get run data for each version
            run1 = self.client.get_run(mv1.run_id)
            run2 = self.client.get_run(mv2.run_id)

            # Extract metrics
            metrics1 = run1.data.metrics
            metrics2 = run2.data.metrics

            # Filter metrics if specified
            if metric_keys:
                metrics1 = {k: v for k, v in metrics1.items() if k in metric_keys}
                metrics2 = {k: v for k, v in metrics2.items() if k in metric_keys}

            # Create comparison DataFrame
            all_metrics = set(metrics1.keys()) | set(metrics2.keys())
            comparison_data = []

            for metric in sorted(all_metrics):
                val1 = metrics1.get(metric)
                val2 = metrics2.get(metric)

                if val1 is not None and val2 is not None:
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else None
                else:
                    diff = None
                    pct_change = None

                comparison_data.append({
                    'metric': metric,
                    f'v{version1}': val1,
                    f'v{version2}': val2,
                    'difference': diff,
                    'pct_change': pct_change
                })

            df = pd.DataFrame(comparison_data)
            logger.info(f"📊 Compared versions {version1} and {version2} of '{model_name}'")
            return df

        except Exception as e:
            logger.error(f"❌ Error comparing model versions: {str(e)}")
            raise

    def auto_promote_model(
        self,
        model_name: str,
        staging_version: str,
        metric_name: str = "accuracy",
        metric_threshold: Optional[float] = None,
        compare_to_production: bool = True,
        higher_is_better: bool = True
    ) -> bool:
        """
        Automatically promote a model based on metric criteria

        Args:
            model_name: Name of the registered model
            staging_version: Version in staging to potentially promote
            metric_name: Metric to evaluate
            metric_threshold: Minimum metric value required for promotion
            compare_to_production: Whether to compare against current production
            higher_is_better: Whether higher metric values are better

        Returns:
            True if model was promoted, False otherwise
        """
        try:
            # Get staging model run
            staging_mv = self.client.get_model_version(model_name, staging_version)
            staging_run = self.client.get_run(staging_mv.run_id)
            staging_metric = staging_run.data.metrics.get(metric_name)

            if staging_metric is None:
                logger.warning(f"⚠️  Metric '{metric_name}' not found for staging version")
                return False

            # Check absolute threshold
            if metric_threshold is not None:
                if higher_is_better and staging_metric < metric_threshold:
                    logger.info(
                        f"❌ Staging model metric {staging_metric:.4f} below threshold {metric_threshold:.4f}"
                    )
                    return False
                elif not higher_is_better and staging_metric > metric_threshold:
                    logger.info(
                        f"❌ Staging model metric {staging_metric:.4f} above threshold {metric_threshold:.4f}"
                    )
                    return False

            # Compare to production if requested
            if compare_to_production:
                prod_version = self.get_latest_model_version(model_name, stage="Production")
                if prod_version:
                    prod_run = self.client.get_run(prod_version['run_id'])
                    prod_metric = prod_run.data.metrics.get(metric_name)

                    if prod_metric is not None:
                        if higher_is_better and staging_metric <= prod_metric:
                            logger.info(
                                f"❌ Staging metric {staging_metric:.4f} not better than "
                                f"production {prod_metric:.4f}"
                            )
                            return False
                        elif not higher_is_better and staging_metric >= prod_metric:
                            logger.info(
                                f"❌ Staging metric {staging_metric:.4f} not better than "
                                f"production {prod_metric:.4f}"
                            )
                            return False

            # Promote model
            self.promote_model(
                model_name=model_name,
                version=staging_version,
                from_stage="Staging",
                to_stage="Production",
                archive_existing=True
            )

            logger.info(
                f"✅ Auto-promoted '{model_name}' v{staging_version} to Production "
                f"({metric_name}: {staging_metric:.4f})"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Error in auto-promotion: {str(e)}")
            return False

    def list_registered_models(self) -> pd.DataFrame:
        """
        List all registered models with summary information

        Returns:
            DataFrame with registered model information
        """
        try:
            models = self.client.search_registered_models()

            if not models:
                logger.info("No registered models found")
                return pd.DataFrame()

            model_data = []
            for model in models:
                all_versions = self.client.search_model_versions(f"name='{model.name}'")

                model_data.append({
                    'name': model.name,
                    'total_versions': len(all_versions),
                    'latest_version': max([int(v.version) for v in all_versions]) if all_versions else None,
                    'creation_time': pd.to_datetime(model.creation_timestamp, unit='ms'),
                    'last_updated': pd.to_datetime(model.last_updated_timestamp, unit='ms'),
                    'description': model.description or 'N/A'
                })

            df = pd.DataFrame(model_data)
            return df.sort_values('last_updated', ascending=False)

        except Exception as e:
            logger.error(f"❌ Error listing registered models: {str(e)}")
            return pd.DataFrame()

    def get_model_version_details(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get detailed information about model version(s)

        Args:
            model_name: Name of the registered model
            version: Specific version (if None, returns all versions)

        Returns:
            DataFrame with version details
        """
        try:
            if version:
                versions = [self.client.get_model_version(model_name, version)]
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")

            if not versions:
                logger.info(f"No versions found for model '{model_name}'")
                return pd.DataFrame()

            version_data = []
            for v in versions:
                try:
                    run = self.client.get_run(v.run_id)
                    metrics = run.data.metrics
                    params = run.data.params
                except Exception as e:
                    logger.warning(f"Could not fetch run data for version {v.version}: {e}")
                    metrics = {}
                    params = {}

                version_data.append({
                    'version': v.version,
                    'stage': v.current_stage,
                    'status': v.status,
                    'run_id': v.run_id[:8] + '...',
                    'created': pd.to_datetime(v.creation_timestamp, unit='ms'),
                    'accuracy': metrics.get('accuracy'),
                    'f1_score': metrics.get('f1_score'),
                    'description': (v.description or '')[:50]
                })

            df = pd.DataFrame(version_data)
            return df.sort_values('version', ascending=False)

        except Exception as e:
            logger.error(f"❌ Error getting model version details: {str(e)}")
            return pd.DataFrame()

    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a specific model version"""
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"🗑️  Deleted '{model_name}' version {version}")
        except Exception as e:
            logger.error(f"❌ Error deleting model version: {str(e)}")
            raise

    def delete_registered_model(self, model_name: str) -> None:
        """Delete a registered model and all its versions"""
        try:
            self.client.delete_registered_model(name=model_name)
            logger.info(f"🗑️  Deleted registered model: '{model_name}'")
        except Exception as e:
            logger.error(f"❌ Error deleting registered model: {str(e)}")
            raise
