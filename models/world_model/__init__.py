# -*- coding: utf-8 -*-
"""
世界模型增强模块
基于《最新任务计划.md》的五大核心组件

模块说明:
- WSM (World State Memory): 时序一致性，降低生成方差
- DCA-FIM (Deformable Cross-Attention): 几何对齐，减少配准误差
- DSC (Differentiable Sensor Consistency): 物理一致性，收紧光谱误差
- WAC-X (Wavelength-Agnostic Cross-band): 频域一致性，高频能量守恒
- Patch Prior Refiner: 生成流形约束，抑制伪影

数学依据:
    统一优化目标 (MAP形式):
    min ||Î - I*||² + λs·R_sens + λg·R_geom + λw·R_wacx + λp·R_patch
    s.t. Î ∈ C(h_t, GeoPos)
    
    其中:
    - R_sens: 可微物理一致性 (DSC)
    - R_geom: 几何一致性 (DCA-FIM)
    - R_wacx: 频域一致性 (WAC-X)
    - R_patch: 生成流形约束 (Patch Prior)
    - C(h_t, GeoPos): 由世界状态h_t和几何编码控制的可行集
"""

__version__ = '1.0.0'
__author__ = 'MambaIR-GPPNN Team'

# 延迟导入，避免循环依赖
def __getattr__(name):
    """延迟导入模块，仅在使用时加载"""
    if name == 'WorldStateMemory':
        from .wsm import WorldStateMemory
        return WorldStateMemory
    elif name == 'DeformableCrossAttention':
        from .dca_fim import DeformableCrossAttention
        return DeformableCrossAttention
    elif name == 'SensorConsistencyLoss':
        from .sensor_loss import SensorConsistencyLoss
        return SensorConsistencyLoss
    elif name == 'WACXLoss':
        from .wacx_loss import WACXLoss
        return WACXLoss
    elif name == 'PatchPriorRefiner':
        from .patch_refiner import PatchPriorRefiner
        return PatchPriorRefiner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'WorldStateMemory',
    'DeformableCrossAttention',
    'SensorConsistencyLoss',
    'WACXLoss',
    'PatchPriorRefiner',
]

