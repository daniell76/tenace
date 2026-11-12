mock_provider "google" {}

run "creates_bucket_and_artifact_registry" {
  command = plan

  variables {
    project_id            = "your-gcp-project-id"
    region                = "us-central1"
    create_tfstate_bucket = true
    tfstate_bucket_name   = "your-gcp-project-id-tfstate"
  }

  assert {
    condition     = length(google_storage_bucket.tfstate) == 1
    error_message = "Expected tfstate GCS bucket resource when create_tfstate_bucket is true."
  }

  assert {
    condition     = google_storage_bucket.tfstate[0].uniform_bucket_level_access == true
    error_message = "Expected uniform_bucket_level_access = true on tfstate bucket."
  }

  assert {
    condition     = google_storage_bucket.tfstate[0].public_access_prevention == "enforced"
    error_message = "Expected public_access_prevention = enforced on tfstate bucket."
  }

  assert {
    condition     = google_storage_bucket.tfstate[0].force_destroy == false
    error_message = "Expected force_destroy = false on tfstate bucket."
  }

  assert {
    condition     = google_storage_bucket.tfstate[0].versioning[0].enabled == true
    error_message = "Expected object versioning enabled on tfstate bucket."
  }

  assert {
    condition     = google_artifact_registry_repository.containers.format == "DOCKER"
    error_message = "Expected DOCKER Artifact Registry repository."
  }

  assert {
    condition     = output.artifact_registry_repository_url == "us-central1-docker.pkg.dev/your-gcp-project-id/tenace"
    error_message = "Unexpected artifact_registry_repository_url output."
  }
}

run "reuses_existing_tfstate_bucket" {
  command = plan

  variables {
    project_id            = "your-gcp-project-id"
    region                = "us-central1"
    create_tfstate_bucket = false
    tfstate_bucket_name   = "your-gcp-project-id-tfstate"
  }

  assert {
    condition     = length(google_storage_bucket.tfstate) == 0
    error_message = "Expected zero tfstate bucket resources when create_tfstate_bucket is false."
  }

  assert {
    condition     = output.tfstate_bucket_name == "your-gcp-project-id-tfstate"
    error_message = "Expected tfstate_bucket_name output to reflect var.tfstate_bucket_name."
  }
}

run "rejects_blank_project_id" {
  command = plan

  variables {
    project_id          = "   "
    tfstate_bucket_name = "your-gcp-project-id-tfstate"
  }

  expect_failures = [
    var.project_id,
  ]
}
