locals {
  base_runtime_env = {
    GOOGLE_CLOUD_PROJECT                 = var.project_id
    TENACE_MODE                          = var.tenace_mode
    TENACE_STRATEGIST_MODEL              = var.strategist_model
    TENACE_STRATEGIST_LOCATION           = var.strategist_location
    TENACE_EVALUATOR_BACKEND             = var.evaluator_backend
    TENACE_DECISIONS_AUTH                = var.decisions_auth
    TENACE_DECISIONS_VIOLATION_THRESHOLD = tostring(var.decisions_violation_threshold)
    TENACE_DDS_WORLDS                    = tostring(var.dds_worlds)
  }

  optional_runtime_env = {
    TENACE_EVALUATOR_MODEL       = var.evaluator_model
    TENACE_EVALUATOR_LOCATION    = var.evaluator_location
    TENACE_EVALUATOR_ENDPOINT_ID = var.evaluator_endpoint_id
  }

  runtime_env_map = merge(
    local.base_runtime_env,
    { for k, v in local.optional_runtime_env : k => v if v != null },
  )

  common_labels = merge(
    {
      app         = "tenace"
      environment = var.environment
      managed_by  = "terraform"
    },
    var.labels,
  )
}

resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "${var.service_name_prefix}-run-${var.environment}"
  display_name = "Tenace Cloud Run (${var.environment})"
}

resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "gemini_api_key" {
  count     = var.gemini_api_key_secret_id != null ? 1 : 0
  project   = var.project_id
  secret_id = var.gemini_api_key_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name                = "${var.service_name_prefix}-api-${var.environment}"
  location            = var.region
  project             = var.project_id
  deletion_protection = var.deletion_protection
  ingress             = "INGRESS_TRAFFIC_ALL"
  labels              = local.common_labels

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "${var.request_timeout_seconds}s"
    max_instance_request_concurrency = var.max_instance_request_concurrency

    scaling {
      min_instance_count = var.min_instance_count
      max_instance_count = var.max_instance_count
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      dynamic "env" {
        for_each = local.runtime_env_map
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = var.gemini_api_key_secret_id != null ? [var.gemini_api_key_secret_id] : []
        content {
          name = "GEMINI_API_KEY"
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      resources {
        limits = {
          cpu    = var.service_cpu
          memory = var.service_memory
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        initial_delay_seconds = 2
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 6
        http_get {
          path = "/healthz"
          port = 8080
        }
      }

      liveness_probe {
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
        http_get {
          path = "/healthz"
          port = 8080
        }
      }
    }
  }

  lifecycle {
    precondition {
      condition     = var.evaluator_endpoint_id == null || var.evaluator_backend == "diffusion-gemma"
      error_message = "evaluator_endpoint_id is only supported when evaluator_backend is 'diffusion-gemma'."
    }

    precondition {
      condition     = var.decisions_auth == "vertex" || var.evaluator_backend == "decisions"
      error_message = "decisions_auth != 'vertex' is only supported when evaluator_backend is 'decisions'."
    }

    precondition {
      condition     = var.decisions_auth != "api-key" || var.gemini_api_key_secret_id != null
      error_message = "gemini_api_key_secret_id is required when decisions_auth is 'api-key'."
    }

    precondition {
      condition     = var.max_instance_count >= var.min_instance_count
      error_message = "max_instance_count must be greater than or equal to min_instance_count."
    }
  }

  depends_on = [
    google_project_iam_member.vertex_user,
    google_project_iam_member.log_writer,
    google_secret_manager_secret_iam_member.gemini_api_key,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count    = var.allow_unauthenticated ? 1 : 0
  project  = google_cloud_run_v2_service.api.project
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_job" "benchmark" {
  count               = var.enable_benchmark_job ? 1 : 0
  name                = "${var.service_name_prefix}-bench-${var.environment}"
  location            = var.region
  project             = var.project_id
  deletion_protection = var.deletion_protection
  labels              = local.common_labels

  template {
    template {
      service_account = google_service_account.runtime.email
      timeout         = "${var.job_timeout_seconds}s"
      max_retries     = var.job_max_retries

      containers {
        image   = var.container_image
        command = ["tenace-match"]
        args    = var.job_args

        dynamic "env" {
          for_each = local.runtime_env_map
          content {
            name  = env.key
            value = env.value
          }
        }

        dynamic "env" {
          for_each = var.gemini_api_key_secret_id != null ? [var.gemini_api_key_secret_id] : []
          content {
            name = "GEMINI_API_KEY"
            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }

        resources {
          limits = {
            cpu    = var.job_cpu
            memory = var.job_memory
          }
        }
      }
    }
  }

  depends_on = [
    google_project_iam_member.vertex_user,
    google_project_iam_member.log_writer,
    google_secret_manager_secret_iam_member.gemini_api_key,
  ]
}
