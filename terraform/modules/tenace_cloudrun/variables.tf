variable "project_id" {
  description = "Google Cloud project ID."
  type        = string

  validation {
    condition     = trimspace(var.project_id) != ""
    error_message = "project_id must be a non-blank string."
  }
}

variable "environment" {
  description = "Deployment environment name (dev, staging, or prod)."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "region" {
  description = "Google Cloud region for Cloud Run Service and Job."
  type        = string
  default     = "us-central1"

  validation {
    condition     = trimspace(var.region) != ""
    error_message = "region must be a non-blank string."
  }
}

variable "service_name_prefix" {
  description = "Lowercase prefix for Cloud Run Service, Job, and Service Account names."
  type        = string
  default     = "tenace"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.service_name_prefix))
    error_message = "service_name_prefix must match ^[a-z][a-z0-9-]{1,20}$."
  }
}

variable "container_image" {
  description = "Full Artifact Registry Docker image URI for Tenace."
  type        = string

  validation {
    condition     = trimspace(var.container_image) != ""
    error_message = "container_image must be a non-blank string."
  }
}

variable "tenace_mode" {
  description = "Tenace execution mode ('vertex' for live models or 'mock' for offline smoke testing)."
  type        = string
  default     = "vertex"

  validation {
    condition     = contains(["vertex", "mock"], var.tenace_mode)
    error_message = "tenace_mode must be 'vertex' or 'mock'."
  }
}

variable "strategist_model" {
  description = "Layer-1 Strategist Vertex AI model identifier."
  type        = string
  default     = "gemini-4-argon"

  validation {
    condition     = trimspace(var.strategist_model) != ""
    error_message = "strategist_model must be a non-blank string."
  }
}

variable "strategist_location" {
  description = "Vertex AI location for the Layer-1 Strategist model."
  type        = string
  default     = "global"

  validation {
    condition     = trimspace(var.strategist_location) != ""
    error_message = "strategist_location must be a non-blank string."
  }
}

variable "evaluator_backend" {
  description = "Layer-2 Tactical Option Evaluator backend ('decisions', 'diffusion-gemma', or 'gemini')."
  type        = string
  default     = "decisions"

  validation {
    condition     = contains(["decisions", "diffusion-gemma", "gemini"], var.evaluator_backend)
    error_message = "evaluator_backend must be one of: decisions, diffusion-gemma, gemini."
  }
}

variable "evaluator_model" {
  description = "Optional Layer-2 Evaluator model override (auto-resolved per backend in Tenace if null)."
  type        = string
  nullable    = true
  default     = null

  validation {
    condition     = var.evaluator_model == null ? true : trimspace(var.evaluator_model) != ""
    error_message = "evaluator_model must not be blank when provided."
  }
}

variable "evaluator_location" {
  description = "Optional Layer-2 Evaluator Vertex AI location override (auto-resolved per backend if null)."
  type        = string
  nullable    = true
  default     = null

  validation {
    condition     = var.evaluator_location == null ? true : trimspace(var.evaluator_location) != ""
    error_message = "evaluator_location must not be blank when provided."
  }
}

variable "evaluator_endpoint_id" {
  description = "Optional dedicated Vertex AI Endpoint ID when evaluator_backend is 'diffusion-gemma'."
  type        = string
  nullable    = true
  default     = null

  validation {
    condition     = var.evaluator_endpoint_id == null ? true : trimspace(var.evaluator_endpoint_id) != ""
    error_message = "evaluator_endpoint_id must not be blank when provided."
  }
}

variable "decisions_auth" {
  description = "Authentication mode for Gemini Decisions API ('vertex' or 'api-key')."
  type        = string
  default     = "vertex"

  validation {
    condition     = contains(["vertex", "api-key"], var.decisions_auth)
    error_message = "decisions_auth must be 'vertex' or 'api-key'."
  }
}

variable "decisions_violation_threshold" {
  description = "Probability threshold strictly in (0, 1) triggering Layer-1 Strategist replanning."
  type        = number
  default     = 0.70

  validation {
    condition     = var.decisions_violation_threshold > 0 && var.decisions_violation_threshold < 1
    error_message = "decisions_violation_threshold must be strictly between 0 and 1."
  }
}

variable "gemini_api_key_secret_id" {
  description = "Secret Manager secret ID storing GEMINI_API_KEY (required when decisions_auth = 'api-key')."
  type        = string
  nullable    = true
  default     = null

  validation {
    condition     = var.gemini_api_key_secret_id == null ? true : trimspace(var.gemini_api_key_secret_id) != ""
    error_message = "gemini_api_key_secret_id must not be blank when provided."
  }
}

variable "dds_worlds" {
  description = "Default number of constrained double-dummy infostate worlds per rollout."
  type        = number
  default     = 16

  validation {
    condition     = var.dds_worlds >= 1 && floor(var.dds_worlds) == var.dds_worlds
    error_message = "dds_worlds must be an integer >= 1."
  }
}

variable "service_cpu" {
  description = "CPU limit per Cloud Run Service instance."
  type        = string
  default     = "2"

  validation {
    condition     = trimspace(var.service_cpu) != ""
    error_message = "service_cpu must be a non-blank string."
  }
}

variable "service_memory" {
  description = "Memory limit per Cloud Run Service instance."
  type        = string
  default     = "2Gi"

  validation {
    condition     = trimspace(var.service_memory) != ""
    error_message = "service_memory must be a non-blank string."
  }
}

variable "min_instance_count" {
  description = "Minimum autoscaled Cloud Run Service instances."
  type        = number
  default     = 0

  validation {
    condition     = var.min_instance_count >= 0 && floor(var.min_instance_count) == var.min_instance_count
    error_message = "min_instance_count must be an integer >= 0."
  }
}

variable "max_instance_count" {
  description = "Maximum autoscaled Cloud Run Service instances."
  type        = number
  default     = 4

  validation {
    condition     = var.max_instance_count >= 1 && floor(var.max_instance_count) == var.max_instance_count
    error_message = "max_instance_count must be an integer >= 1."
  }
}

variable "max_instance_request_concurrency" {
  description = "Maximum concurrent HTTP requests per Cloud Run Service instance."
  type        = number
  default     = 4

  validation {
    condition     = var.max_instance_request_concurrency >= 1 && floor(var.max_instance_request_concurrency) == var.max_instance_request_concurrency
    error_message = "max_instance_request_concurrency must be an integer >= 1."
  }
}

variable "request_timeout_seconds" {
  description = "HTTP request timeout in seconds for Cloud Run Service (1..3600)."
  type        = number
  default     = 300

  validation {
    condition     = var.request_timeout_seconds >= 1 && var.request_timeout_seconds <= 3600 && floor(var.request_timeout_seconds) == var.request_timeout_seconds
    error_message = "request_timeout_seconds must be an integer between 1 and 3600."
  }
}

variable "allow_unauthenticated" {
  description = "Whether to grant roles/run.invoker to allUsers on the Cloud Run Service."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Whether Cloud Run deletion protection is enabled on the Service and Job."
  type        = bool
  default     = false
}

variable "enable_benchmark_job" {
  description = "Whether to provision the companion Cloud Run Job for WBF Duplicate IMP benchmarks."
  type        = bool
  default     = true
}

variable "job_cpu" {
  description = "CPU limit for the Cloud Run benchmark Job container."
  type        = string
  default     = "2"

  validation {
    condition     = trimspace(var.job_cpu) != ""
    error_message = "job_cpu must be a non-blank string."
  }
}

variable "job_memory" {
  description = "Memory limit for the Cloud Run benchmark Job container."
  type        = string
  default     = "2Gi"

  validation {
    condition     = trimspace(var.job_memory) != ""
    error_message = "job_memory must be a non-blank string."
  }
}

variable "job_timeout_seconds" {
  description = "Task execution timeout in seconds for the Cloud Run benchmark Job (60..86400)."
  type        = number
  default     = 3600

  validation {
    condition     = var.job_timeout_seconds >= 60 && var.job_timeout_seconds <= 86400 && floor(var.job_timeout_seconds) == var.job_timeout_seconds
    error_message = "job_timeout_seconds must be an integer between 60 and 86400."
  }
}

variable "job_max_retries" {
  description = "Maximum task retries for the Cloud Run benchmark Job."
  type        = number
  default     = 0

  validation {
    condition     = var.job_max_retries >= 0 && floor(var.job_max_retries) == var.job_max_retries
    error_message = "job_max_retries must be an integer >= 0."
  }
}

variable "job_args" {
  description = "Default CLI arguments passed to `tenace-match` in the Cloud Run benchmark Job."
  type        = list(string)
  default     = ["--boards", "4", "--dds-worlds", "16"]
}

variable "labels" {
  description = "Additional resource labels merged onto Cloud Run resources."
  type        = map(string)
  default     = {}
}
