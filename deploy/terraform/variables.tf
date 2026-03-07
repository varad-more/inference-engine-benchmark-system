# =============================================================================
# variables.tf — Input variables for the benchmark infrastructure
# =============================================================================

# ---------------------------------------------------------------------------
# AWS Region & credentials
# ---------------------------------------------------------------------------

variable "aws_region" {
  type        = string
  description = "AWS region to deploy into."
  default     = "us-east-1"
}

# ---------------------------------------------------------------------------
# Deployment topology
# ---------------------------------------------------------------------------

variable "deployment_mode" {
  type        = string
  description = <<-EOT
    "single" — one g5.2xlarge runs both vLLM + SGLang + dashboard.
               Engines share the GPU and run sequentially (cheapest).
    "multi"  — dedicated GPU instance per engine + a separate t3.medium
               for the dashboard. True isolation for fair benchmarking.
  EOT
  default     = "single"

  validation {
    condition     = contains(["single", "multi"], var.deployment_mode)
    error_message = "deployment_mode must be \"single\" or \"multi\"."
  }
}

# ---------------------------------------------------------------------------
# Instance types
# ---------------------------------------------------------------------------

variable "instance_type_gpu" {
  type        = string
  description = <<-EOT
    GPU instance type for the inference engines.
    Recommended options:
      g4dn.xlarge   — 1× T4 (16 GB),  4 vCPU, 16 GB RAM  — cheapest dev tier
      g5.2xlarge    — 1× A10G (24 GB), 8 vCPU, 32 GB RAM  — recommended
      p3.2xlarge    — 1× V100 (16 GB), 8 vCPU, 61 GB RAM  — high memory
      g5.12xlarge   — 4× A10G (96 GB),48 vCPU,192 GB RAM  — multi-GPU
  EOT
  default     = "g5.2xlarge"
}

variable "instance_type_dashboard" {
  type        = string
  description = "CPU instance type for the dashboard (multi mode only)."
  default     = "t3.medium"
}

# ---------------------------------------------------------------------------
# Access & credentials
# ---------------------------------------------------------------------------

variable "key_pair_name" {
  type        = string
  description = "Name of an existing EC2 key pair for SSH access."
}

variable "your_ip_cidr" {
  type        = string
  description = <<-EOT
    Your public IP in CIDR notation (e.g. \"1.2.3.4/32\").
    SSH and dashboard access are restricted to this IP.
    Find yours: curl -s https://checkip.amazonaws.com
  EOT
}

variable "hf_token" {
  type        = string
  description = "HuggingFace Hub token for downloading gated models."
  sensitive   = true
  default     = ""
}

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

variable "model_id" {
  type        = string
  description = "HuggingFace model ID to load in both engines."
  default     = "Qwen/Qwen2.5-1.5B-Instruct"
}

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

variable "volume_size_gb" {
  type        = number
  description = <<-EOT
    EBS root volume size in GB.
    Model cache + Docker images + results easily consume 50-80 GB.
    100 GB is a safe default; increase for larger models.
  EOT
  default     = 100
}

variable "volume_type" {
  type        = string
  description = "EBS volume type. gp3 is faster and cheaper than gp2."
  default     = "gp3"
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the dedicated VPC."
  default     = "10.42.0.0/16"
}

variable "subnet_cidr" {
  type        = string
  description = "CIDR block for the public subnet."
  default     = "10.42.1.0/24"
}

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

variable "project_name" {
  type        = string
  description = "Tag prefix applied to all AWS resources."
  default     = "llm-benchmark"
}

variable "git_repo_url" {
  type        = string
  description = "Git repository URL cloned onto each instance during bootstrap."
  default     = "https://github.com/your-org/inference-engine-benchmark-system.git"
}
