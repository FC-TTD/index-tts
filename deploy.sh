#!/bin/bash
set -e

# 默认值
DOCKER_MODE=false
SWARM_MODE=false
ANSIBLE_DIR="./ansible"

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --docker | docker)
      DOCKER_MODE=true
      shift
      ;;
    --swarm | swarm)
      SWARM_MODE=true
      shift
      ;;
    *)
      echo "Usage: $0 [--pkg] [--docker] [--swarm]"
      echo "Examples:"
      echo "  $0           # 使用Docker完整CI/CD流程部署"
      echo "  $0 --pkg      # 仅打包项目"
      echo "  $0 --docker   # 部署 Docker 镜像"
      echo "  $0 --swarm    # 部署 Swarm 服务"
      exit 1
      ;;
  esac
done

# 设置标签
TAGS="docker"
if [ "$SWARM_MODE" = true ]; then
  TAGS="swarm"
  echo "部署 Swarm 服务..."
else
  echo "部署 Docker 镜像..."
fi

# 执行 Ansible playbook
CMD="ansible-playbook -i $ANSIBLE_DIR/inventory.yml $ANSIBLE_DIR/site.yml --tags $TAGS"
echo 执行命令: $CMD
ANSIBLE_STDOUT_CALLBACK=debug $CMD
