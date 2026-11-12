variable "project_id" {
  description = "Google Cloud project ID hosting Tenace infrastructure."
  type        = string

  validation {
    condition     = trimspace(var.project_id) != ""
    error_message = "project_id must be a non-blank string."
  }
}

variable "region" {
  description = "Default Google Cloud region for Artifact Registry and provider resources."
  type        = string
  default     = "us-central1"

  validation {
    condition     = trimspace(var.region) != ""
    error_message = "region must be a non-blank string."
  }
}

variable "create_tfstate_bucket" {
  description = "Set to false when reusing an existing versioned GCS bucket for Terraform remote state."
  type        = bool
  default     = true
}

variable "tfstate_bucket_name" {
  description = "Globally unique GCS bucket name for Terraform remote state."
  type        = string

  validation {
    condition     = trimspace(var.tfstate_bucket_name) != ""
    error_message = "tfstate_bucket_name must be a non-blank string."
  }
}

variable "tfstate_bucket_location" {
  description = "GCS location for the Terraform remote state bucket (when created)."
  type        = string
  default     = "US-CENTRAL1"

  validation {
    condition     = trimspace(var.tfstate_bucket_location) != ""
    error_message = "tfstate_bucket_location must be a non-blank string."
  }
}

variable "artifact_registry_repo_id" {
  description = "Artifact Registry Docker repository ID."
  type        = string
  default     = "tenace"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,61}[a-z0-9]$", var.artifact_registry_repo_id))
    error_message = "artifact_registry_repo_id must match ^[a-z][a-z0-9-]{1,61}[a-z0-9]$."
  }
}
