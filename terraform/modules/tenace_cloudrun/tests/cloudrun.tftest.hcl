mock_provider "google" {}

run "default_decisions_service_and_job" {
  command = plan

  variables {
    project_id      = "your-gcp-project-id"
    environment     = "dev"
    region          = "us-central1"
    container_image = "us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:latest"
  }

  assert {
    condition     = google_service_account.runtime.account_id == "tenace-run-dev"
    error_message = "Unexpected runtime service account_id."
  }

  assert {
    condition     = google_cloud_run_v2_service.api.name == "tenace-api-dev"
    error_message = "Unexpected Cloud Run Service name."
  }

  assert {
    condition     = alltrue([for e in google_cloud_run_v2_service.api.template[0].containers[0].env : e.name != "PORT"])
    error_message = "PORT is reserved by Cloud Run v2 and must not be injected in container env."
  }

  assert {
    condition = (
      { for e in google_cloud_run_v2_service.api.template[0].containers[0].env : e.name => e.value if e.value != null }
    )["TENACE_EVALUATOR_BACKEND"] == "decisions"
    error_message = "Expected default TENACE_EVALUATOR_BACKEND = decisions."
  }

  assert {
    condition = (
      { for e in google_cloud_run_v2_service.api.template[0].containers[0].env : e.name => e.value if e.value != null }
    )["TENACE_STRATEGIST_MODEL"] == "gemini-4-argon"
    error_message = "Expected default TENACE_STRATEGIST_MODEL = gemini-4-argon."
  }

  assert {
    condition     = length(google_cloud_run_v2_service_iam_member.public_invoker) == 0
    error_message = "Expected zero public IAM bindings when allow_unauthenticated = false."
  }

  assert {
    condition     = length(google_cloud_run_v2_job.benchmark) == 1
    error_message = "Expected benchmark Cloud Run Job to be enabled by default."
  }

  assert {
    condition     = google_cloud_run_v2_job.benchmark[0].template[0].template[0].containers[0].command == tolist(["tenace-match"])
    error_message = "Expected benchmark Job container command to be ['tenace-match']."
  }

  assert {
    condition     = google_cloud_run_v2_job.benchmark[0].template[0].template[0].containers[0].args == tolist(["--boards", "4"])
    error_message = "Expected default benchmark Job args to be ['--boards', '4'] so TENACE_DDS_WORLDS governs worlds."
  }
}

run "diffusion_gemma_endpoint_and_public_invoker" {
  command = plan

  variables {
    project_id            = "your-gcp-project-id"
    environment           = "staging"
    region                = "us-central1"
    container_image       = "us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:latest"
    evaluator_backend     = "diffusion-gemma"
    evaluator_location    = "us-central1"
    evaluator_endpoint_id = "1234567890"
    allow_unauthenticated = true
  }

  assert {
    condition = (
      { for e in google_cloud_run_v2_service.api.template[0].containers[0].env : e.name => e.value if e.value != null }
    )["TENACE_EVALUATOR_BACKEND"] == "diffusion-gemma"
    error_message = "Expected TENACE_EVALUATOR_BACKEND = diffusion-gemma."
  }

  assert {
    condition = (
      { for e in google_cloud_run_v2_service.api.template[0].containers[0].env : e.name => e.value if e.value != null }
    )["TENACE_EVALUATOR_ENDPOINT_ID"] == "1234567890"
    error_message = "Expected TENACE_EVALUATOR_ENDPOINT_ID = 1234567890."
  }

  assert {
    condition     = length(google_cloud_run_v2_service_iam_member.public_invoker) == 1
    error_message = "Expected allUsers run.invoker binding when allow_unauthenticated = true."
  }
}

run "decisions_api_key_with_secret_manager" {
  command = plan

  variables {
    project_id               = "your-gcp-project-id"
    environment              = "prod"
    region                   = "us-central1"
    container_image          = "us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:v0.1.0"
    evaluator_backend        = "decisions"
    decisions_auth           = "api-key"
    gemini_api_key_secret_id = "tenace-gemini-api-key"
  }

  assert {
    condition     = length(google_secret_manager_secret_iam_member.gemini_api_key) == 1
    error_message = "Expected Secret Manager secretAccessor IAM binding for gemini_api_key_secret_id."
  }

  assert {
    condition     = anytrue([for e in google_cloud_run_v2_service.api.template[0].containers[0].env : e.name == "GEMINI_API_KEY"])
    error_message = "Expected GEMINI_API_KEY secret env var on Cloud Run Service container."
  }

  assert {
    condition     = anytrue([for e in google_cloud_run_v2_job.benchmark[0].template[0].template[0].containers[0].env : e.name == "GEMINI_API_KEY"])
    error_message = "Expected GEMINI_API_KEY secret env var on Cloud Run benchmark Job container."
  }
}

run "rejects_endpoint_id_with_decisions_backend" {
  command = plan

  variables {
    project_id            = "your-gcp-project-id"
    environment           = "dev"
    container_image       = "us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:latest"
    evaluator_backend     = "decisions"
    evaluator_endpoint_id = "1234567890"
  }

  expect_failures = [
    google_cloud_run_v2_service.api,
  ]
}

run "rejects_api_key_auth_without_secret_id" {
  command = plan

  variables {
    project_id        = "your-gcp-project-id"
    environment       = "dev"
    container_image   = "us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:latest"
    evaluator_backend = "decisions"
    decisions_auth    = "api-key"
  }

  expect_failures = [
    google_cloud_run_v2_service.api,
  ]
}

run "rejects_max_instances_below_min_instances" {
  command = plan

  variables {
    project_id         = "your-gcp-project-id"
    environment        = "dev"
    container_image    = "us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:latest"
    min_instance_count = 3
    max_instance_count = 2
  }

  expect_failures = [
    google_cloud_run_v2_service.api,
  ]
}
