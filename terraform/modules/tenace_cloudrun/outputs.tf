output "service_name" {
  description = "Cloud Run v2 Service name."
  value       = google_cloud_run_v2_service.api.name
}

output "service_uri" {
  description = "Cloud Run v2 Service HTTPS URI."
  value       = google_cloud_run_v2_service.api.uri
}

output "service_account_email" {
  description = "Dedicated least-privilege Cloud Run runtime Service Account email."
  value       = google_service_account.runtime.email
}

output "benchmark_job_name" {
  description = "Cloud Run v2 benchmark Job name (or null when enable_benchmark_job is false)."
  value       = var.enable_benchmark_job ? google_cloud_run_v2_job.benchmark[0].name : null
}
