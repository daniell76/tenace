output "service_name" {
  description = "Prod Cloud Run v2 Service name."
  value       = module.tenace.service_name
}

output "service_uri" {
  description = "Prod Cloud Run v2 Service HTTPS URI."
  value       = module.tenace.service_uri
}

output "service_account_email" {
  description = "Prod Cloud Run runtime Service Account email."
  value       = module.tenace.service_account_email
}

output "benchmark_job_name" {
  description = "Prod Cloud Run v2 benchmark Job name."
  value       = module.tenace.benchmark_job_name
}
