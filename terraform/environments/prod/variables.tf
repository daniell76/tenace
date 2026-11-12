variable "project_id" {
  description = "Google Cloud project ID for the prod environment."
  type        = string
}

variable "region" {
  description = "Google Cloud region for Cloud Run Service and Job."
  type        = string
  default     = "us-central1"
}

variable "service_name_prefix" {
  description = "Lowercase prefix for Cloud Run resources."
  type        = string
  default     = "tenace"
}

variable "container_image" {
  description = "Full Artifact Registry Docker image URI for Tenace."
  type        = string
}

variable "tenace_mode" {
  description = "Tenace execution mode ('vertex' or 'mock')."
  type        = string
  default     = "vertex"
}

variable "strategist_model" {
  description = "Layer-1 Strategist Vertex AI model identifier."
  type        = string
  default     = "gemini-4-argon"
}

variable "strategist_location" {
  description = "Vertex AI location for the Layer-1 Strategist model."
  type        = string
  default     = "global"
}

variable "evaluator_backend" {
  description = "Layer-2 Tactical Option Evaluator backend ('decisions', 'diffusion-gemma', or 'gemini')."
  type        = string
  default     = "decisions"
}

variable "evaluator_model" {
  description = "Optional Layer-2 Evaluator model override."
  type        = string
  nullable    = true
  default     = null
}

variable "evaluator_location" {
  description = "Optional Layer-2 Evaluator Vertex AI location override."
  type        = string
  nullable    = true
  default     = null
}

variable "evaluator_endpoint_id" {
  description = "Optional dedicated Vertex AI Endpoint ID for DiffusionGemma."
  type        = string
  nullable    = true
  default     = null
}

variable "decisions_auth" {
  description = "Authentication mode for Gemini Decisions API ('vertex' or 'api-key')."
  type        = string
  default     = "vertex"
}

variable "decisions_violation_threshold" {
  description = "Probability threshold in (0, 1) triggering Layer-1 replanning."
  type        = number
  default     = 0.70
}

variable "gemini_api_key_secret_id" {
  description = "Secret Manager secret ID storing GEMINI_API_KEY when decisions_auth = 'api-key'."
  type        = string
  nullable    = true
  default     = null
}

variable "dds_worlds" {
  description = "Default constrained double-dummy infostate worlds per rollout."
  type        = number
  default     = 16
}

variable "service_cpu" {
  description = "CPU limit per Cloud Run Service instance."
  type        = string
  default     = "2"
}

variable "service_memory" {
  description = "Memory limit per Cloud Run Service instance."
  type        = string
  default     = "2Gi"
}

variable "min_instance_count" {
  description = "Minimum autoscaled Cloud Run Service instances."
  type        = number
  default     = 0
}

variable "max_instance_count" {
  description = "Maximum autoscaled Cloud Run Service instances in prod."
  type        = number
  default     = 10
}

variable "max_instance_request_concurrency" {
  description = "Maximum concurrent HTTP requests per Cloud Run Service instance."
  type        = number
  default     = 4
}

variable "request_timeout_seconds" {
  description = "HTTP request timeout in seconds for Cloud Run Service."
  type        = number
  default     = 300
}

variable "allow_unauthenticated" {
  description = "Whether to grant public roles/run.invoker access on the Cloud Run Service."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Whether deletion protection is enabled on prod Cloud Run resources."
  type        = bool
  default     = true
}

variable "enable_benchmark_job" {
  description = "Whether to provision the companion Cloud Run benchmark Job."
  type        = bool
  default     = true
}

variable "job_cpu" {
  description = "CPU limit for the Cloud Run benchmark Job container."
  type        = string
  default     = "2"
}

variable "job_memory" {
  description = "Memory limit for the Cloud Run benchmark Job container."
  type        = string
  default     = "2Gi"
}

variable "job_timeout_seconds" {
  description = "Task execution timeout in seconds for the Cloud Run benchmark Job."
  type        = number
  default     = 3600
}

variable "job_max_retries" {
  description = "Maximum task retries for the Cloud Run benchmark Job."
  type        = number
  default     = 0
}

variable "job_args" {
  description = "CLI arguments passed to tenace-match in the Cloud Run benchmark Job."
  type        = list(string)
  default     = ["--boards", "8", "--dds-worlds", "16"]
}

variable "labels" {
  description = "Additional resource labels merged onto prod Cloud Run resources."
  type        = map(string)
  default     = {}
}
