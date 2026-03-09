"""
Ingest Manager - 文件指纹检测 + Parse 结果缓存

解决的问题：
1. 文件指纹检测：检测文件是否变化，决定要不要重新 ingest
2. Parse 缓存：即使需要 ingest，也不重复调用昂贵的 LlamaParse API

使用方式：
    manager = IngestManager()

    # 检查是否需要重新 ingest
    status = manager.check_ingest_status(pdf_files, col_name, collection_exists, collection_count)

    if status.skip:
        # Collection 存在且文件没变，跳过
        pass
    else:
        # 有文件变化或 collection 不存在，需要处理
        for filepath in pdf_files:
            parsed = manager.get_or_parse(filepath, parse_fn)
        # ingest 完成后记录
        manager.mark_ingested_files(col_name, pdf_files)
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class IngestStatus:
    """Ingest 状态检查结果"""
    skip: bool = False                      # 是否可以完全跳过
    reason: str = ""                        # 原因说明
    changed_files: list[str] = field(default_factory=list)   # 变化的文件（新增或修改）
    deleted_files: list[str] = field(default_factory=list)   # 删除的文件
    unchanged_files: list[str] = field(default_factory=list) # 未变化的文件


class IngestManager:
    """
    管理 RAG ingest 的缓存和增量更新
    
    功能：
    1. 文件指纹检测 - 通过 MD5 hash 检测文件是否变化
    2. Parse 缓存 - 缓存 LlamaParse 的结果，避免重复调用 API
    3. Manifest 管理 - 记录已 ingest 的文件和版本
    """
    
    def __init__(self, 
                 manifest_path: str = None,
                 cache_dir: str = None):
        """
        Args:
            manifest_path: manifest 文件路径，默认在项目根目录的 .ingest_manifest.json
            cache_dir: parse 缓存目录，默认在项目根目录的 .parse_cache/
        """
        project_root = Path(__file__).resolve().parents[2]
        
        self.manifest_path = manifest_path or str(project_root / ".ingest_manifest.json")
        self.cache_dir = Path(cache_dir or str(project_root / ".parse_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._manifest = self._load_manifest()
    
    # =========================================================================
    # Manifest 管理 - 记录已 ingest 的文件指纹
    # =========================================================================
    
    def _load_manifest(self) -> dict:
        """加载 manifest 文件"""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_manifest(self):
        """保存 manifest 文件"""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)
    
    def _file_hash(self, filepath: str) -> str:
        """计算文件的 MD5 指纹"""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            # 分块读取，避免大文件内存问题
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    # =========================================================================
    # 文件指纹检测 - 决定要不要重新 ingest
    # =========================================================================
    
    def check_ingest_status(self,
                            pdf_files: list[str],
                            col_name: str,
                            collection_exists: bool,
                            collection_count: int = 0) -> IngestStatus:
        """
        检查是否需要重新 ingest

        Args:
            pdf_files: 要检查的 PDF 文件路径列表（已过滤好的）
            col_name: Milvus collection 名称
            collection_exists: collection 是否已存在
            collection_count: collection 中的 entity 数量

        Returns:
            IngestStatus 包含是否跳过、变化的文件列表等信息
        """
        if not pdf_files:
            return IngestStatus(
                skip=False,
                reason="no PDF files provided",
            )

        # 获取上次 ingest 时的文件指纹
        cached_files = self._manifest.get(col_name, {}).get("files", {})

        # 比较文件指纹
        changed_files = []
        unchanged_files = []

        for filepath in pdf_files:
            fname = os.path.basename(filepath)
            current_hash = self._file_hash(filepath)
            if cached_files.get(fname) != current_hash:
                changed_files.append(filepath)
            else:
                unchanged_files.append(filepath)

        # 检查是否有文件被删除（只针对本次传入的文件名范围）
        current_fnames = {os.path.basename(fp) for fp in pdf_files}
        deleted_files = [
            fname for fname in cached_files.keys()
            if fname not in current_fnames
        ]

        # 决定是否跳过
        if collection_exists and collection_count > 0 and not changed_files and not deleted_files:
            return IngestStatus(
                skip=True,
                reason=f"collection exists with {collection_count} chunks, no file changes detected",
                unchanged_files=unchanged_files,
            )

        reason_parts = []
        if not collection_exists or collection_count == 0:
            reason_parts.append("collection is empty or missing")
        if changed_files:
            reason_parts.append(f"{len(changed_files)} file(s) changed/added")
        if deleted_files:
            reason_parts.append(f"{len(deleted_files)} file(s) deleted")

        return IngestStatus(
            skip=False,
            reason="; ".join(reason_parts) if reason_parts else "initial ingest",
            changed_files=changed_files,
            deleted_files=deleted_files,
            unchanged_files=unchanged_files,
        )
    
    def mark_ingested_files(self, col_name: str, file_paths: list[str]):
        """
        标记该 collection 已完成 ingest，记录实际处理的文件指纹

        Args:
            col_name: Milvus collection 名称
            file_paths: 实际 ingest 的文件路径列表

        在 ingest 完成后调用
        """
        files = {}
        for filepath in file_paths:
            if os.path.exists(filepath):
                files[os.path.basename(filepath)] = self._file_hash(filepath)

        self._manifest[col_name] = {
            "files": files,
            "last_ingested": self._get_timestamp(),
        }
        self._save_manifest()
    
    @staticmethod
    def _get_timestamp() -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # =========================================================================
    # Parse 缓存 - 避免重复调用 LlamaParse API
    # =========================================================================
    
    def _get_cache_paths(self, filepath: str) -> tuple[Path, Path]:
        """获取缓存文件路径（key = 文件内容 MD5，与路径无关，支持跨机器共享）"""
        filename = os.path.basename(filepath)
        cache_subdir = self.cache_dir / self._file_hash(filepath)[:8]
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        parsed_cache = cache_subdir / f"{filename}.parsed.json"
        hash_cache = cache_subdir / f"{filename}.md5"
        
        return parsed_cache, hash_cache
    
    def get_cached_parse(self, filepath: str) -> str | None:
        """
        获取缓存的 parse 结果
        
        Args:
            filepath: PDF 文件路径
        
        Returns:
            缓存的 parse 结果（文本），如果缓存无效则返回 None
        """
        if not os.path.exists(filepath):
            return None
        
        parsed_cache, hash_cache = self._get_cache_paths(filepath)
        
        # 检查缓存是否存在
        if not parsed_cache.exists() or not hash_cache.exists():
            return None
        
        # 检查文件指纹是否匹配
        current_hash = self._file_hash(filepath)
        try:
            cached_hash = hash_cache.read_text().strip()
        except IOError:
            return None
        
        if current_hash != cached_hash:
            # 文件已变化，缓存无效
            return None
        
        # 读取缓存的 parse 结果
        try:
            with open(parsed_cache, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                return cache_data.get("text")
        except (json.JSONDecodeError, IOError):
            return None
    
    def save_parse_cache(self, filepath: str, parsed_text: str):
        """
        保存 parse 结果到缓存
        
        Args:
            filepath: PDF 文件路径
            parsed_text: LlamaParse 返回的文本
        """
        if not os.path.exists(filepath):
            return
        
        parsed_cache, hash_cache = self._get_cache_paths(filepath)
        
        # 保存文件指纹
        current_hash = self._file_hash(filepath)
        hash_cache.write_text(current_hash)
        
        # 保存 parse 结果
        cache_data = {
            "text": parsed_text,
            "source_file": os.path.basename(filepath),
            "cached_at": self._get_timestamp()
        }
        with open(parsed_cache, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    def get_or_parse(self, 
                     filepath: str, 
                     parse_fn: Callable[[str], str],
                     verbose: bool = True) -> str:
        """
        获取 parse 结果，优先使用缓存
        
        Args:
            filepath: PDF 文件路径
            parse_fn: 解析函数，接收文件路径，返回解析后的文本
            verbose: 是否打印日志
        
        Returns:
            解析后的文本
        """
        filename = os.path.basename(filepath)
        
        # 尝试使用缓存
        cached = self.get_cached_parse(filepath)
        if cached is not None:
            if verbose:
                print(f"  [cache] {filename}: using cached parse result")
            return cached
        
        # 缓存未命中，调用 parse
        if verbose:
            print(f"  [parse] {filename}: calling LlamaParse API...")
        
        parsed_text = parse_fn(filepath)
        
        # 保存到缓存
        self.save_parse_cache(filepath, parsed_text)
        if verbose:
            print(f"  [cache] {filename}: saved to cache")
        
        return parsed_text
    
    def clear_cache(self, col_name: str = None):
        """
        清除缓存
        
        Args:
            col_name: 如果指定，只清除该 collection 的 manifest 记录
                      如果为 None，清除所有 manifest 记录
        """
        if col_name:
            if col_name in self._manifest:
                del self._manifest[col_name]
                self._save_manifest()
        else:
            self._manifest = {}
            self._save_manifest()
        
        # 注意：不自动清除 parse_cache 目录，因为 parse 缓存可以跨 collection 复用
    
    def clear_parse_cache(self):
        """清除所有 parse 缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)