#!/bin/bash
set -e

# 默认值
CI_MODE=false
CD_MODE=false
ANSIBLE_DIR="./ansible"

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --ci)
      CI_MODE=true
      shift
      ;;
    --cd)
      CD_MODE=true
      shift
      ;;
    *)
      echo "Usage: $0 [--ci] [--cd]"
      echo "Examples:"
      echo "  $0           # 使用Docker完整CI/CD流程部署"
      echo "  $0 --ci      # 仅构建 Docker 镜像"
      echo "  $0 --cd      # 仅部署 Docker 容器"
      exit 1
      ;;
  esac
done

# 设置标签
TAGS="docker"
if [ "$CI_MODE" = true ] && [ "$CD_MODE" = false ]; then
  TAGS="$TAGS,ci"
  echo "仅构建镜像 (docker,ci)..."
elif [ "$CI_MODE" = false ] && [ "$CD_MODE" = true ]; then
  TAGS="$TAGS,cd"
  echo "仅部署服务 (docker,cd)..."
else
  TAGS="$TAGS,ci,cd"
  echo "使用CI/CD流水线部署 (docker,ci,cd)..."
fi

# 执行 Ansible playbook
ANSIBLE_STDOUT_CALLBACK=debug ansible-playbook $ANSIBLE_DIR/site.yml --tags "$TAGS"
