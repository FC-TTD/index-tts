#!/bin/bash
set -e

# 默认值
DOCKER_MODE=false
SWARM_MODE=false
PUB_MODE=false
ANSIBLE_DIR="./ansible"

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --docker | docker)
      DOCKER_MODE=true
      shift
      ;;
    --swarm | swarm | api)
      SWARM_MODE=true
      shift
      ;;
    --pub | pypi)
      PUB_MODE=true
      shift
      ;;
    *)
      echo "Usage: $0 [--pkg] [--docker] [--swarm] [--pub]"
      echo "Examples:"
      echo "  $0             # 使用Docker完整CI/CD流程部署"
      echo "  $0 --pkg       # 仅打包项目"
      echo "  $0 --docker    # 部署 Docker 镜像"
      echo "  $0 --swarm|api # 部署 Swarm 服务"
      echo "  $0 --pub|pypi  # 发布 PyPI 包"
      exit 1
      ;;
  esac
done

# 设置标签和playbook
PLAYBOOK="$ANSIBLE_DIR/site.yml"
TAGS="docker"
if [ "$PUB_MODE" = true ]; then
  PLAYBOOK="packages/pub.yml"
  TAGS=""
  echo "发布 PyPI 包..."
elif [ "$SWARM_MODE" = true ]; then
  TAGS="swarm"
  echo "部署 Swarm 服务..."
else
  echo "部署 Docker 镜像..."
fi

# 执行 Ansible playbook
if [ "$PUB_MODE" = true ]; then
  CMD="ansible-playbook -i localhost, $PLAYBOOK --connection=local"
else
  CMD="ansible-playbook -i $ANSIBLE_DIR/inventory.yml $PLAYBOOK --tags $TAGS"
fi
echo 执行命令: $CMD
ANSIBLE_STDOUT_CALLBACK=debug $CMD
