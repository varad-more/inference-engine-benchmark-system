# =============================================================================
# main.tf — vLLM vs SGLang Benchmark System on AWS
#
# Supports two deployment modes:
#   "single" — one GPU instance hosts both engines + dashboard (cheapest)
#   "multi"  — dedicated GPU instance per engine + CPU dashboard instance
#              (true isolation, recommended for fair benchmarking)
#
# Usage:
#   terraform init
#   terraform plan -var="key_pair_name=my-key" -var="your_ip_cidr=1.2.3.4/32"
#   terraform apply ...
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      ManagedBy   = "Terraform"
      Benchmark   = "vllm-vs-sglang"
    }
  }
}

# =============================================================================
# AMI — AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
# Pre-installed: NVIDIA drivers, CUDA toolkit, Docker
# =============================================================================

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# =============================================================================
# Networking
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-public-subnet" }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${var.project_name}-rt" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# =============================================================================
# Security Groups
# =============================================================================

# --- Shared group: lets instances talk to each other on engine ports -----------
resource "aws_security_group" "internal" {
  name        = "${var.project_name}-internal"
  description = "Allow inter-service traffic between benchmark nodes"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "vLLM API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "SGLang API"
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound (model downloads, apt, etc.)"
  }

  tags = { Name = "${var.project_name}-internal-sg" }
}

# --- GPU instance SG: SSH + engine ports (engines only reachable internally) --
resource "aws_security_group" "gpu" {
  name        = "${var.project_name}-gpu"
  description = "GPU inference engine instances"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH from your IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  # Engine ports — only accessible from within the internal SG (dashboard → engines)
  ingress {
    description     = "vLLM (internal)"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.internal.id]
  }

  ingress {
    description     = "SGLang (internal)"
    from_port       = 8001
    to_port         = 8001
    protocol        = "tcp"
    security_groups = [aws_security_group.internal.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-gpu-sg" }
}

# --- Dashboard SG: SSH + port 3000 from your IP only -------------------------
resource "aws_security_group" "dashboard" {
  name        = "${var.project_name}-dashboard"
  description = "Dashboard / benchmark CLI instance"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH from your IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  ingress {
    description = "Dashboard UI from your IP"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-dashboard-sg" }
}

# =============================================================================
# IAM role — allows SSM Session Manager as SSH fallback
# =============================================================================

resource "aws_iam_role" "ec2_ssm" {
  name = "${var.project_name}-ec2-ssm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2_ssm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_ssm" {
  name = "${var.project_name}-ec2-ssm-profile"
  role = aws_iam_role.ec2_ssm.name
}

# =============================================================================
# EBS root volume configuration (shared across all instances)
# =============================================================================

locals {
  root_block_device = {
    volume_type           = var.volume_type
    volume_size           = var.volume_size_gb
    delete_on_termination = true
    encrypted             = true
  }

  # Template variables injected into every user-data script
  common_vars = {
    HF_TOKEN    = var.hf_token
    MODEL_ID    = var.model_id
    GIT_REPO    = var.git_repo_url
    PROJECT     = var.project_name
  }
}

# =============================================================================
# OPTION A — Single Instance (deployment_mode = "single")
# Both engines + dashboard on one GPU node
# =============================================================================

resource "aws_instance" "single" {
  count = var.deployment_mode == "single" ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type_gpu
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public.id
  iam_instance_profile   = aws_iam_instance_profile.ec2_ssm.name

  vpc_security_group_ids = [
    aws_security_group.gpu.id,
    aws_security_group.dashboard.id,
    aws_security_group.internal.id,
  ]

  root_block_device {
    volume_type           = local.root_block_device.volume_type
    volume_size           = local.root_block_device.volume_size
    delete_on_termination = local.root_block_device.delete_on_termination
    encrypted             = local.root_block_device.encrypted
    tags                  = { Name = "${var.project_name}-single-root" }
  }

  user_data = templatefile("${path.module}/user_data/gpu_instance.sh", merge(local.common_vars, {
    MODE          = "single"
    VLLM_HOST     = "localhost"
    SGLANG_HOST   = "localhost"
  }))

  tags = { Name = "${var.project_name}-single" }

  lifecycle {
    ignore_changes = [ami] # Prevent replacement when new DL AMI releases
  }
}

# =============================================================================
# OPTION B — Multi Instance (deployment_mode = "multi")
# Dedicated GPU per engine + CPU dashboard
# =============================================================================

# --- vLLM GPU instance -------------------------------------------------------
resource "aws_instance" "vllm" {
  count = var.deployment_mode == "multi" ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type_gpu
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public.id
  iam_instance_profile   = aws_iam_instance_profile.ec2_ssm.name

  vpc_security_group_ids = [
    aws_security_group.gpu.id,
    aws_security_group.internal.id,
  ]

  root_block_device {
    volume_type           = local.root_block_device.volume_type
    volume_size           = local.root_block_device.volume_size
    delete_on_termination = local.root_block_device.delete_on_termination
    encrypted             = local.root_block_device.encrypted
    tags                  = { Name = "${var.project_name}-vllm-root" }
  }

  user_data = templatefile("${path.module}/user_data/gpu_instance.sh", merge(local.common_vars, {
    MODE        = "vllm_only"
    VLLM_HOST   = "localhost"
    SGLANG_HOST = ""
  }))

  tags = { Name = "${var.project_name}-vllm" }

  lifecycle { ignore_changes = [ami] }
}

# --- SGLang GPU instance -----------------------------------------------------
resource "aws_instance" "sglang" {
  count = var.deployment_mode == "multi" ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type_gpu
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public.id
  iam_instance_profile   = aws_iam_instance_profile.ec2_ssm.name

  vpc_security_group_ids = [
    aws_security_group.gpu.id,
    aws_security_group.internal.id,
  ]

  root_block_device {
    volume_type           = local.root_block_device.volume_type
    volume_size           = local.root_block_device.volume_size
    delete_on_termination = local.root_block_device.delete_on_termination
    encrypted             = local.root_block_device.encrypted
    tags                  = { Name = "${var.project_name}-sglang-root" }
  }

  user_data = templatefile("${path.module}/user_data/gpu_instance.sh", merge(local.common_vars, {
    MODE        = "sglang_only"
    VLLM_HOST   = ""
    SGLANG_HOST = "localhost"
  }))

  tags = { Name = "${var.project_name}-sglang" }

  lifecycle { ignore_changes = [ami] }
}

# --- Dashboard / CLI CPU instance --------------------------------------------
resource "aws_instance" "dashboard" {
  count = var.deployment_mode == "multi" ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type_dashboard
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public.id
  iam_instance_profile   = aws_iam_instance_profile.ec2_ssm.name

  vpc_security_group_ids = [
    aws_security_group.dashboard.id,
    aws_security_group.internal.id,
  ]

  root_block_device {
    volume_type           = local.root_block_device.volume_type
    volume_size           = 30 # Dashboard needs much less space
    delete_on_termination = local.root_block_device.delete_on_termination
    encrypted             = local.root_block_device.encrypted
    tags                  = { Name = "${var.project_name}-dashboard-root" }
  }

  # dashboard.sh receives the private IPs of the engine nodes
  user_data = templatefile("${path.module}/user_data/dashboard.sh", merge(local.common_vars, {
    VLLM_HOST   = var.deployment_mode == "multi" ? aws_instance.vllm[0].private_ip : "localhost"
    SGLANG_HOST = var.deployment_mode == "multi" ? aws_instance.sglang[0].private_ip : "localhost"
  }))

  tags = { Name = "${var.project_name}-dashboard" }

  lifecycle { ignore_changes = [ami] }

  # Dashboard waits for engine instances to be created first so it
  # can resolve their private IPs during user-data templating
  depends_on = [aws_instance.vllm, aws_instance.sglang]
}

# =============================================================================
# Elastic IPs — stable public addresses
# =============================================================================

resource "aws_eip" "single" {
  count    = var.deployment_mode == "single" ? 1 : 0
  instance = aws_instance.single[0].id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-single-eip" }
}

resource "aws_eip" "vllm" {
  count    = var.deployment_mode == "multi" ? 1 : 0
  instance = aws_instance.vllm[0].id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-vllm-eip" }
}

resource "aws_eip" "sglang" {
  count    = var.deployment_mode == "multi" ? 1 : 0
  instance = aws_instance.sglang[0].id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-sglang-eip" }
}

resource "aws_eip" "dashboard" {
  count    = var.deployment_mode == "multi" ? 1 : 0
  instance = aws_instance.dashboard[0].id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-dashboard-eip" }
}
