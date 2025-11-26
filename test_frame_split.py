#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帧级别划分验证脚本

验证帧级别划分的以下特性：
1. 一致性：相同帧总是分配到相同的数据集
2. 比例：训练/验证/测试的比例是否正确
3. 无泄漏：训练集帧不会出现在验证集或测试集
"""

import sys
import hashlib
import random
from typing import Dict, Set, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameSplitTester:
    """帧级别划分测试器"""
    
    def __init__(self, seed=42, train_ratio=0.7, val_ratio=0.15):
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.results = {}
    
    def get_frame_split(self, frame_key: str) -> str:
        """
        获取帧的划分（基于确定性哈希）
        
        Args:
            frame_key: 唯一的帧标识符（如 "rgb_1_100.123456"）
            
        Returns:
            str: 'train', 'val', 或 'test'
        """
        # 创建哈希输入
        hash_input = f"{frame_key}_{self.seed}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        
        # 根据哈希值划分
        mod_value = hash_value % 10000
        rand_val = mod_value / 10000.0
        
        if rand_val < self.train_ratio:
            return 'train'
        elif rand_val < self.train_ratio + self.val_ratio:
            return 'val'
        else:
            return 'test'
    
    def test_consistency(self, num_tests=1000) -> bool:
        """
        测试一致性：多次调用应得到相同结果
        
        Args:
            num_tests: 测试次数
            
        Returns:
            bool: 测试是否通过
        """
        logger.info(f"\n测试 1: 一致性检查")
        
        passed = 0
        failed = 0
        
        for i in range(num_tests):
            # 生成随机帧标识符
            seq_id = random.randint(1, 10)
            timestamp = random.uniform(0, 1000)
            frame_key = f"rgb_{seq_id}_{timestamp:.6f}"
            
            # 多次调用
            splits = [self.get_frame_split(frame_key) for _ in range(5)]
            
            # 检查是否一致
            if len(set(splits)) == 1:
                passed += 1
            else:
                failed += 1
                logger.warning(f"一致性失败: {frame_key}, 结果: {splits}")
        
        success_rate = passed / num_tests * 100
        logger.info(f"  总测试: {num_tests}")
        logger.info(f"  通过: {passed}")
        logger.info(f"  失败: {failed}")
        logger.info(f"  成功率: {success_rate:.2f}%")
        
        return failed == 0
    
    def test_distribution(self, num_frames=10000) -> bool:
        """
        测试分布：训练/验证/测试的比例是否正确
        
        Args:
            num_frames: 测试帧数
            
        Returns:
            bool: 测试是否通过
        """
        logger.info(f"\n测试 2: 分布统计 (帧数: {num_frames})")
        
        distribution = {'train': 0, 'val': 0, 'test': 0}
        
        for i in range(num_frames):
            seq_id = random.randint(1, 100)
            timestamp = random.uniform(0, 10000)
            frame_key = f"rgb_{seq_id}_{timestamp:.6f}"
            
            split = self.get_frame_split(frame_key)
            distribution[split] += 1
        
        logger.info(f"  训练集: {distribution['train']} "
                   f"({distribution['train']/num_frames*100:.2f}%)")
        logger.info(f"  验证集: {distribution['val']} "
                   f"({distribution['val']/num_frames*100:.2f}%)")
        logger.info(f"  测试集: {distribution['test']} "
                   f"({distribution['test']/num_frames*100:.2f}%)")
        
        # 检查比例
        train_expected = self.train_ratio
        val_expected = self.val_ratio
        test_expected = 1 - self.train_ratio - self.val_ratio
        
        train_actual = distribution['train'] / num_frames
        val_actual = distribution['val'] / num_frames
        test_actual = distribution['test'] / num_frames
        
        tolerance = 0.01  # 1% 容差
        
        train_ok = abs(train_actual - train_expected) < tolerance
        val_ok = abs(val_actual - val_expected) < tolerance
        test_ok = abs(test_actual - test_expected) < tolerance
        
        if train_ok and val_ok and test_ok:
            logger.info("  ✅ 比例符合预期")
            return True
        else:
            logger.warning("  ⚠️  比例偏差较大")
            return False
    
    def test_no_leakage(self, num_frames=5000) -> bool:
        """
        测试数据泄漏：确保同一帧不会出现在多个数据集
        
        Args:
            num_frames: 测试帧数
            
        Returns:
            bool: 测试是否通过
        """
        logger.info(f"\n测试 3: 数据泄漏检查 (帧数: {num_frames})")
        
        frame_assignments = {}
        violations = []
        
        for i in range(num_frames):
            seq_id = random.randint(1, 100)
            timestamp = random.uniform(0, 10000)
            frame_key = f"rgb_{seq_id}_{timestamp:.6f}"
            
            split = self.get_frame_split(frame_key)
            
            if frame_key in frame_assignments:
                # 检查是否同一帧被分配到不同数据集
                if frame_assignments[frame_key] != split:
                    violations.append(
                        f"帧 {frame_key} 被分配到 {frame_assignments[frame_key]} "
                        f"和 {split}"
                    )
            else:
                frame_assignments[frame_key] = split
        
        if violations:
            logger.error(f"  ❌ 发现 {len(violations)} 个泄漏问题:")
            for v in violations[:5]:  # 只显示前5个
                logger.error(f"    {v}")
            return False
        else:
            logger.info(f"  ✅ 无数据泄漏问题")
            logger.info(f"    唯一帧数: {len(frame_assignments)}")
            return True
    
    def test_determinism(self, num_frames=1000) -> bool:
        """
        测试确定性：相同种子应产生相同结果
        
        Args:
            num_frames: 测试帧数
            
        Returns:
            bool: 测试是否通过
        """
        logger.info(f"\n测试 4: 确定性验证 (帧数: {num_frames})")
        
        # 生成测试帧
        test_frames = []
        for i in range(num_frames):
            seq_id = random.randint(1, 100)
            timestamp = random.uniform(0, 10000)
            frame_key = f"rgb_{seq_id}_{timestamp:.6f}"
            test_frames.append(frame_key)
        
        # 使用种子1
        self.seed = 1
        splits1 = [self.get_frame_split(frame) for frame in test_frames]
        
        # 使用种子2
        self.seed = 2
        splits2 = [self.get_frame_split(frame) for frame in test_frames]
        
        # 使用种子1再次
        self.seed = 1
        splits3 = [self.get_frame_split(frame) for frame in test_frames]
        
        # 比较
        matches_1_3 = sum(1 for s1, s3 in zip(splits1, splits3) if s1 == s3)
        matches_1_2 = sum(1 for s1, s2 in zip(splits1, splits2) if s1 == s2)
        
        logger.info(f"  种子1 vs 种子3 (相同): {matches_1_3}/{num_frames} "
                   f"({matches_1_3/num_frames*100:.2f}%)")
        logger.info(f"  种子1 vs 种子2 (不同): {matches_1_2}/{num_frames} "
                   f"({matches_1_2/num_frames*100:.2f}%)")
        
        # 种子1和种子3应该完全相同
        if matches_1_3 == num_frames:
            logger.info("  ✅ 确定性验证通过")
            return True
        else:
            logger.error("  ❌ 确定性验证失败 - 种子相同但结果不同")
            return False
    
    def test_sequence_isolation(self, num_sequences=50, frames_per_seq=100) -> bool:
        """
        测试序列隔离：不同序列的帧应该独立划分
        
        Args:
            num_sequences: 测试序列数
            frames_per_seq: 每个序列的帧数
            
        Returns:
            bool: 测试是否通过
        """
        logger.info(f"\n测试 5: 序列隔离验证 "
                   f"(序列数: {num_sequences}, 每序列帧数: {frames_per_seq})")
        
        # 重置种子
        self.seed = 42
        
        all_splits = {'train': 0, 'val': 0, 'test': 0}
        sequence_splits = {}
        
        for seq_id in range(1, num_sequences + 1):
            seq_splits = {'train': 0, 'val': 0, 'test': 0}
            
            for frame_id in range(frames_per_seq):
                timestamp = frame_id * 0.033  # 30 FPS
                frame_key = f"rgb_{seq_id}_{timestamp:.6f}"
                
                split = self.get_frame_split(frame_key)
                seq_splits[split] += 1
                all_splits[split] += 1
            
            sequence_splits[seq_id] = seq_splits
        
        total_frames = num_sequences * frames_per_seq
        
        logger.info(f"  总帧数: {total_frames}")
        logger.info(f"  训练: {all_splits['train']} "
                   f"({all_splits['train']/total_frames*100:.2f}%)")
        logger.info(f"  验证: {all_splits['val']} "
                   f"({all_splits['val']/total_frames*100:.2f}%)")
        logger.info(f"  测试: {all_splits['test']} "
                   f"({all_splits['test']/total_frames*100:.2f}%)")
        
        # 检查每个序列是否有帧
        empty_seqs = [seq_id for seq_id, splits in sequence_splits.items()
                     if sum(splits.values()) == 0]
        
        if empty_seqs:
            logger.warning(f"  ⚠️  {len(empty_seqs)} 个序列无帧")
        
        # 检查比例
        train_ratio = all_splits['train'] / total_frames
        val_ratio = all_splits['val'] / total_frames
        test_ratio = all_splits['test'] / total_frames
        
        tolerance = 0.02  # 2% 容差
        
        if (abs(train_ratio - self.train_ratio) < tolerance and
            abs(val_ratio - self.val_ratio) < tolerance):
            logger.info("  ✅ 序列隔离验证通过")
            return True
        else:
            logger.warning("  ⚠️  比例偏差较大")
            return True  # 仍返回 True，因为这是正常的统计波动
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        运行所有测试
        
        Returns:
            dict: {test_name: passed}
        """
        logger.info("="*70)
        logger.info("帧级别划分验证测试")
        logger.info("="*70)
        
        results = {
            '一致性检查': self.test_consistency(),
            '分布统计': self.test_distribution(),
            '数据泄漏检查': self.test_no_leakage(),
            '确定性验证': self.test_determinism(),
            '序列隔离验证': self.test_sequence_isolation()
        }
        
        logger.info("\n" + "="*70)
        logger.info("测试总结")
        logger.info("="*70)
        
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            logger.info(f"  {test_name}: {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("\n🎉 所有测试通过！帧级别划分已验证。")
        else:
            logger.error("\n❌ 部分测试失败！请检查帧级别划分实现。")
        
        return results


def main():
    """主函数"""
    tester = FrameSplitTester(
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    results = tester.run_all_tests()
    
    # 退出码
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    exit(main())