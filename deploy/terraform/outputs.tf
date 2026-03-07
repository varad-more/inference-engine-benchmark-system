# =============================================================================
# outputs.tf — Actionable connection details after terraform apply
# =============================================================================

# ---------------------------------------------------------------------------
# Helpers — resolve the correct public IPs for each mode
# ---------------------------------------------------------------------------

locals {
  # Single mode
  single_ip = var.deployment_mode == "single" ? (
    length(aws_eip.single) > 0 ? aws_eip.single[0].public_ip : ""
  ) : ""

  # Multi mode — engine IPs
  vllm_ip      = var.deployment_mode == "multi" ? (length(aws_eip.vllm) > 0 ? aws_eip.vllm[0].public_ip : "") : ""
  sglang_ip    = var.deployment_mode == "multi" ? (length(aws_eip.sglang) > 0 ? aws_eip.sglang[0].public_ip : "") : ""
  dash_ip      = var.deployment_mode == "multi" ? (length(aws_eip.dashboard) > 0 ? aws_eip.dashboard[0].public_ip : "") : ""

  # Effective dashboard IP for both modes
  dashboard_ip = var.deployment_mode == "single" ? local.single_ip : local.dash_ip

  # Effective engine IPs for CLI commands (single → localhost; multi → engine EIPs)
  effective_vllm_host   = var.deployment_mode == "single" ? "localhost" : local.vllm_ip
  effective_sglang_host = var.deployment_mode == "single" ? "localhost" : local.sglang_ip
}

# ---------------------------------------------------------------------------
# Instance IDs
# ---------------------------------------------------------------------------

output "instance_ids" {
  description = "Map of role → EC2 instance ID."
  value = var.deployment_mode == "single" ? {
    single = length(aws_instance.single) > 0 ? aws_instance.single[0].id : ""
  } : {
    vllm      = length(aws_instance.vllm) > 0 ? aws_instance.vllm[0].id : ""
    sglang    = length(aws_instance.sglang) > 0 ? aws_instance.sglang[0].id : ""
    dashboard = length(aws_instance.dashboard) > 0 ? aws_instance.dashboard[0].id : ""
  }
}

# ---------------------------------------------------------------------------
# Public IPs
# ---------------------------------------------------------------------------

output "public_ips" {
  description = "Public Elastic IP addresses."
  value = var.deployment_mode == "single" ? {
    single = local.single_ip
  } : {
    vllm      = local.vllm_ip
    sglang    = local.sglang_ip
    dashboard = local.dash_ip
  }
}

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

output "dashboard_url" {
  description = "URL to open the benchmark dashboard in your browser."
  value       = "http://${local.dashboard_ip}:3000"
}

# ---------------------------------------------------------------------------
# SSH commands
# ---------------------------------------------------------------------------

output "ssh_commands" {
  description = "Ready-to-paste SSH commands for each instance."
  value = var.deployment_mode == "single" ? {
    single = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${local.single_ip}"
  } : {
    vllm      = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${local.vllm_ip}"
    sglang    = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${local.sglang_ip}"
    dashboard = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${local.dash_ip}"
  }
}

# ---------------------------------------------------------------------------
# Benchmark CLI commands (run from the single/dashboard instance)
# ---------------------------------------------------------------------------

output "benchmark_commands" {
  description = "Example CLI commands to run once the engines are healthy."
  value = {
    health = join(" ", [
      "python run_experiment.py health",
      "--vllm-host ${local.effective_vllm_host}",
      "--sglang-host ${local.effective_sglang_host}",
    ])

    run_latency = join(" ", [
      "python run_experiment.py run",
      "--scenario single_request_latency",
      "--engines vllm,sglang",
      "--vllm-host ${local.effective_vllm_host}",
      "--sglang-host ${local.effective_sglang_host}",
    ])

    run_throughput = join(" ", [
      "python run_experiment.py run",
      "--scenario throughput_ramp",
      "--engines vllm,sglang",
      "--vllm-host ${local.effective_vllm_host}",
      "--sglang-host ${local.effective_sglang_host}",
    ])

    compare = join(" ", [
      "python run_experiment.py compare",
      "--scenario prefix_sharing_benefit",
      "--vllm-host ${local.effective_vllm_host}",
      "--sglang-host ${local.effective_sglang_host}",
    ])

    report = "python run_experiment.py report --output /tmp/report.html"

    scp_report = "scp -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${local.dashboard_ip}:/home/ubuntu/benchmark/report.html ./report.html"
  }
}

# ---------------------------------------------------------------------------
# AMI used
# ---------------------------------------------------------------------------

output "ami_used" {
  description = "Deep Learning AMI selected for GPU instances."
  value = {
    id   = data.aws_ami.deep_learning.id
    name = data.aws_ami.deep_learning.name
  }
}

# ---------------------------------------------------------------------------
# Cost estimate reminder
# ---------------------------------------------------------------------------

output "cost_reminder" {
  description = "Approximate hourly cost at us-east-1 on-demand pricing (2025)."
  value = var.deployment_mode == "single" ? (
    "~$1.21/hr  (1× ${var.instance_type_gpu}). Stop the instance when idle to save cost."
  ) : (
    "~$2.46/hr  (2× ${var.instance_type_gpu} + 1× ${var.instance_type_dashboard}). Terminate when done."
  )
}
