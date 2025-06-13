"""
PostgreSQL 依赖修复脚本
用于解决 LangGraph PostgresSaver 的导入问题
"""

import subprocess
import sys
import os


def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误: {e.stderr.strip() if e.stderr else str(e)}")
        return False


def fix_postgres_dependencies():
    """修复PostgreSQL依赖问题"""
    print("🚀 开始修复 PostgreSQL 依赖问题...")
    
    fixes_applied = []
    
    # 1. 尝试安装 psycopg2-binary
    if run_command("pip install psycopg2-binary", "安装 psycopg2-binary"):
        fixes_applied.append("安装了 psycopg2-binary")
    
    # 2. 如果上面失败，尝试安装 psycopg[binary]
    else:
        if run_command("pip install 'psycopg[binary]'", "安装 psycopg[binary]"):
            fixes_applied.append("安装了 psycopg[binary]")
        else:
            # 3. 最后尝试纯Python版本
            if run_command("pip install 'psycopg[pool]'", "安装 psycopg[pool] (纯Python版本)"):
                fixes_applied.append("安装了 psycopg[pool] (纯Python版本)")
    
    # 4. 验证LangGraph PostgresSaver是否可用
    print("\n🔍 验证 PostgresSaver 是否可用...")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        print("✅ PostgresSaver 导入成功！")
        fixes_applied.append("PostgresSaver 可用")
    except ImportError as e:
        print(f"⚠️ PostgresSaver 仍然不可用: {e}")
        print("💡 系统将自动使用 MemorySaver 作为备选方案")
        fixes_applied.append("将使用 MemorySaver 作为备选")
    
    # 5. 显示修复结果
    print(f"\n📋 修复结果:")
    if fixes_applied:
        for fix in fixes_applied:
            print(f"  ✅ {fix}")
    else:
        print("  ⚠️ 没有应用任何修复")
    
    return len(fixes_applied) > 0


def check_system_info():
    """检查系统信息"""
    print("🖥️ 系统信息:")
    print(f"  Python 版本: {sys.version}")
    print(f"  操作系统: {os.name}")
    
    if os.name == 'nt':  # Windows
        print("  📝 注意: 在Windows上，推荐使用 psycopg2-binary")
    
    # 检查已安装的相关包
    try:
        import pkg_resources
        installed_packages = [pkg.project_name for pkg in pkg_resources.working_set]
        
        postgres_packages = [pkg for pkg in installed_packages if 'psycopg' in pkg.lower()]
        if postgres_packages:
            print(f"  已安装的PostgreSQL包: {', '.join(postgres_packages)}")
        else:
            print("  未检测到PostgreSQL包")
            
        langgraph_packages = [pkg for pkg in installed_packages if 'langgraph' in pkg.lower()]
        if langgraph_packages:
            print(f"  已安装的LangGraph包: {', '.join(langgraph_packages)}")
            
    except Exception as e:
        print(f"  无法检查已安装包: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("PostgreSQL 依赖修复工具")
    print("=" * 60)
    
    check_system_info()
    
    print(f"\n🎯 目标: 修复 'no pq wrapper available' 错误")
    
    if fix_postgres_dependencies():
        print(f"\n🎉 修复完成！请重新启动应用程序。")
    else:
        print(f"\n⚠️ 修复可能未完全成功。")
        print(f"💡 建议:")
        print(f"  1. 手动运行: pip install psycopg2-binary")
        print(f"  2. 或者设置环境变量: USE_POSTGRES_CHECKPOINTER=false")
        print(f"  3. 系统将自动使用 MemorySaver 作为备选方案")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main() 