output "service_name" {
  description = "Dev Cloud Run v2 Service name."
  value       = module.tenace.service_name
}

output "service_uri" {
  description = "Dev Cloud Run v2 Service HTTPS URI."
  value       = module.tenace.service_uri
}

output "service_account_email" {
  description = "Dev Cloud Run runtime Service Account email."
  value       = module.tenace.service_account_email
}

output "benchmark_job_name" {
  description = "Dev Cloud Run v2 benchmark Job name."
  value       = module.tenace.benchmark_job_name
}
