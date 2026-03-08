#!/usr/bin/env python3
"""
缓存管理 CLI 工具

使用方式:
    # 查看缓存状态
    python src/cache_cli.py status
    
    # 清除特定 collection 的 manifest（下次启动会重新检测）
    python src/cache_cli.py clear --collection nio_ec6
    
    # 清除所有 manifest
    python src/cache_cli.py clear --all-manifest
    
    # 清除所有 parse 缓存（下次 ingest 会重新调用 LlamaParse）
    python src/cache_cli.py clear --all-parse
    
    # 强制重新 ingest 特定车型
    python src/cache_cli.py reingest --model EC6
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from storage.ingest_manager import IngestManager
import config


def cmd_status(args):
    """显示缓存状态"""
    manager = IngestManager()
    
    print("=" * 60)
    print("RAG Ingest Cache Status")
    print("=" * 60)
    
    # Manifest 状态
    print("\n📋 Manifest 文件:")
    print(f"   路径: {manager.manifest_path}")
    if os.path.exists(manager.manifest_path):
        with open(manager.manifest_path) as f:
            manifest = json.load(f)
        print(f"   Collections: {len(manifest)}")
        for col_name, data in manifest.items():
            files = data.get("files", {})
            last_ingested = data.get("last_ingested", "unknown")
            print(f"   - {col_name}: {len(files)} file(s), last ingested: {last_ingested}")
    else:
        print("   (不存在)")
    
    # Parse 缓存状态
    print("\n📦 Parse 缓存:")
    print(f"   路径: {manager.cache_dir}")
    if manager.cache_dir.exists():
        cache_files = list(manager.cache_dir.rglob("*.parsed.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"   缓存文件数: {len(cache_files)}")
        print(f"   总大小: {total_size / 1024 / 1024:.2f} MB")
        if cache_files:
            print("   缓存的文件:")
            for cf in cache_files:
                print(f"     - {cf.name.replace('.parsed.json', '')}")
    else:
        print("   (不存在)")
    
    # Milvus 状态
    print("\n🗄️ Milvus 数据库:")
    print(f"   路径: {config.MILVUS_URI}")
    if os.path.exists(config.MILVUS_URI):
        size = os.path.getsize(config.MILVUS_URI) / 1024 / 1024
        print(f"   大小: {size:.2f} MB")
    else:
        print("   (不存在)")
    
    # 数据目录状态
    print("\n📁 数据目录:")
    print(f"   路径: {config.DATA_ROOT}")
    for model in config.NIO_CAR_MODELS:
        pdf_path = os.path.join(config.DATA_ROOT, f"{model}.pdf")
        col_name = f"nio_{model.lower()}"
        
        if os.path.isfile(pdf_path):
            # 检查是否在 manifest 中
            cached = manager._manifest.get(col_name, {}).get("files", {})
            current_hash = manager._file_hash(pdf_path)
            cached_hash = cached.get(f"{model}.pdf")
            
            if cached_hash is None:
                status_icon = "🆕"  # 未 ingest
                status_text = "未 ingest"
            elif cached_hash != current_hash:
                status_icon = "🔄"  # 文件已修改
                status_text = "文件已修改"
            else:
                status_icon = "✅"  # 已同步
                status_text = "已同步"
            
            size_mb = os.path.getsize(pdf_path) / 1024 / 1024
            print(f"   {status_icon} {model}.pdf ({size_mb:.1f} MB) - {status_text}")
        else:
            print(f"   ❌ {model}.pdf - 文件不存在")
    
    print()


def cmd_clear(args):
    """清除缓存"""
    manager = IngestManager()
    
    if args.collection:
        print(f"清除 collection '{args.collection}' 的 manifest...")
        manager.clear_cache(args.collection)
        print("完成！")
    
    elif args.all_manifest:
        print("清除所有 manifest...")
        manager.clear_cache()
        print("完成！")
    
    elif args.all_parse:
        print("清除所有 parse 缓存...")
        manager.clear_parse_cache()
        print("完成！")
    
    else:
        print("请指定要清除的内容: --collection, --all-manifest, 或 --all-parse")
        sys.exit(1)


def cmd_reingest(args):
    """强制重新 ingest"""
    from rag.pipeline import ingest
    
    model = args.model.upper()
    pdf_path = os.path.join(config.DATA_ROOT, f"{model}.pdf")
    col_name = f"nio_{model.lower()}"
    
    if not os.path.isfile(pdf_path):
        print(f"错误: PDF 文件不存在: {pdf_path}")
        sys.exit(1)
    
    print(f"强制重新 ingest {model}...")
    ctx = ingest(
        data_dir=config.DATA_ROOT,
        uri=config.MILVUS_URI,
        col_name=col_name,
        force=True,
        file_filter=f"{model}.pdf"
    )
    print(f"完成！共 {ctx.store.col.num_entities} chunks")


def main():
    parser = argparse.ArgumentParser(description="RAG Ingest 缓存管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # status 命令
    status_parser = subparsers.add_parser("status", help="显示缓存状态")
    status_parser.set_defaults(func=cmd_status)
    
    # clear 命令
    clear_parser = subparsers.add_parser("clear", help="清除缓存")
    clear_parser.add_argument("--collection", type=str, help="清除特定 collection 的 manifest")
    clear_parser.add_argument("--all-manifest", action="store_true", help="清除所有 manifest")
    clear_parser.add_argument("--all-parse", action="store_true", help="清除所有 parse 缓存")
    clear_parser.set_defaults(func=cmd_clear)
    
    # reingest 命令
    reingest_parser = subparsers.add_parser("reingest", help="强制重新 ingest")
    reingest_parser.add_argument("--model", type=str, required=True, help="车型名称 (如 EC6)")
    reingest_parser.set_defaults(func=cmd_reingest)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()