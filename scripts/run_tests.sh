#!/bin/bash

# 脚本说明：运行项目测试

# 设置默认值
TEST_TYPE="all"
COVERAGE=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --unit)
      TEST_TYPE="unit"
      shift
      ;;
    --integration)
      TEST_TYPE="integration"
      shift
      ;;
    --all)
      TEST_TYPE="all"
      shift
      ;;
    --coverage)
      COVERAGE=1
      shift
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 准备测试命令
if [ $COVERAGE -eq 1 ]; then
  TEST_CMD="pytest --cov=app"
else
  TEST_CMD="pytest"
fi

# 根据测试类型执行不同的测试
case $TEST_TYPE in
  "unit")
    echo "运行单元测试..."
    $TEST_CMD -m "unit" tests/
    ;;
  "integration")
    echo "运行集成测试..."
    $TEST_CMD -m "integration" tests/
    ;;
  "all")
    echo "运行所有测试..."
    $TEST_CMD tests/
    ;;
esac

# 如果有覆盖率报告，生成HTML报告
if [ $COVERAGE -eq 1 ]; then
  echo "生成覆盖率报告..."
  python -m coverage html
  echo "覆盖率报告已生成到 htmlcov/index.html"
fi

echo "测试完成！" 