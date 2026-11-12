output "tfstate_bucket_name" {
  description = "GCS bucket name used for Terraform remote state."
  value       = var.create_tfstate_bucket ? google_storage_bucket.tfstate[0].name : var.tfstate_bucket_name
}

output "artifact_registry_repository_url" {
  description = "Artifact Registry Docker repository base URL."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}
