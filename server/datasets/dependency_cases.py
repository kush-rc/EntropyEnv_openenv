# server/datasets/dependency_cases.py
# Ground truth cases for PyTorch Migration Time-Machine tasks.
# Covers: deprecated API detection, version conflict resolution, graph-break fixing.

DEPENDENCY_CASES = {
    'dep_easy': [
        {
            'case_id': 'dep_easy_001',
            'task_subtype': 'flag',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'expected_outdated_packages': ['torch'],
            'expected_deprecated_api': 'torch.autograd.Variable',
            'replacement': 'plain tensor (remove Variable wrapper)',
            'code_snippet': '''import torch
from torch.autograd import Variable

x = Variable(torch.randn(3, 4), requires_grad=True)
y = Variable(torch.randn(3, 4))
z = x + y''',
            'task_description': 'Identify outdated PyTorch packages and deprecated APIs in this legacy training script.',
        },
        {
            'case_id': 'dep_easy_002',
            'task_subtype': 'flag',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'expected_outdated_packages': ['torch'],
            'expected_deprecated_api': 'tensor.data.numpy()',
            'replacement': 'tensor.detach().numpy()',
            'code_snippet': '''import torch

model = torch.nn.Linear(10, 5)
x = torch.randn(1, 10)
output = model(x)
result = output.data.numpy()  # deprecated''',
            'task_description': 'Find deprecated tensor conversion API in this code.',
        },
        {
            'case_id': 'dep_easy_003',
            'task_subtype': 'flag',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'expected_outdated_packages': ['torch'],
            'expected_deprecated_api': 'model.cuda()',
            'replacement': 'model.to(device)',
            'code_snippet': '''import torch

model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)
model.cuda()  # deprecated device placement
x = torch.randn(1, 784).cuda()''',
            'task_description': 'Detect deprecated device placement API in this model code.',
        },
        {
            'case_id': 'dep_easy_004',
            'task_subtype': 'flag',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'expected_outdated_packages': ['torch'],
            'expected_deprecated_api': 'torch.onnx.export',
            'replacement': 'torch.onnx.dynamo_export',
            'code_snippet': '''import torch

model = torch.nn.Linear(10, 5)
dummy = torch.randn(1, 10)
torch.onnx.export(model, dummy, "model.onnx",
                  opset_version=11)''',
            'task_description': 'Find the deprecated ONNX export API in this code.',
        },
        {
            'case_id': 'dep_easy_005',
            'task_subtype': 'flag',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'expected_outdated_packages': ['torch'],
            'expected_deprecated_api': 'torch.nn.DataParallel',
            'replacement': 'torch.nn.parallel.DistributedDataParallel or FSDP',
            'code_snippet': '''import torch
import torch.nn as nn

model = nn.Linear(100, 10)
model = nn.DataParallel(model)  # deprecated
model.cuda()''',
            'task_description': 'Find deprecated parallelism API in this training code.',
        },
    ],
    'dep_medium': [
        {
            'case_id': 'dep_medium_001',
            'task_subtype': 'resolve',
            'completion_threshold': 0.75,
            'max_steps': 6,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['resolve_conflict']},
            'conflict_packages': ['torch', 'numpy'],
            'compatibility_matrix': {
                'torch': {
                    '2.1.0': {'numpy': '>=1.24,<2.0'},
                    '2.0.0': {'numpy': '>=1.22,<1.26'},
                    '1.13.0': {'numpy': '>=1.19,<1.25'},
                },
                'numpy': {
                    '1.26.0': {},
                    '1.24.0': {},
                    '1.22.0': {},
                    '1.19.0': {},
                    '1.16.0': {},
                },
            },
            'requirements': {'torch': '1.9.0', 'numpy': '1.16.0'},
            'code_snippet': '''# requirements.txt
torch==1.9.0
numpy==1.16.0
torchvision==0.10.0''',
            'task_description': 'Resolve the version conflict between torch and numpy. Find compatible versions using the compatibility matrix.',
        },
        {
            'case_id': 'dep_medium_002',
            'task_subtype': 'resolve',
            'completion_threshold': 0.75,
            'max_steps': 6,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['resolve_conflict']},
            'conflict_packages': ['torch', 'numpy', 'torchvision'],
            'compatibility_matrix': {
                'torch': {
                    '2.2.0': {'numpy': '>=1.24,<2.0', 'torchvision': '>=0.17'},
                    '2.1.0': {'numpy': '>=1.24,<2.0', 'torchvision': '>=0.16'},
                    '2.0.0': {'numpy': '>=1.22,<1.26', 'torchvision': '>=0.15'},
                },
                'numpy': {
                    '1.26.0': {},
                    '1.24.0': {},
                    '1.22.0': {},
                },
                'torchvision': {
                    '0.17.0': {'torch': '>=2.2'},
                    '0.16.0': {'torch': '>=2.1'},
                    '0.15.0': {'torch': '>=2.0'},
                },
            },
            'requirements': {'torch': '1.12.0', 'numpy': '1.21.0', 'torchvision': '0.13.0'},
            'code_snippet': '''# requirements.txt
torch==1.12.0
numpy==1.21.0
torchvision==0.13.0
# CUDA 11.7''',
            'task_description': 'Resolve three-way conflict between PyTorch, NumPy, and TorchVision.',
        },
        {
            'case_id': 'dep_medium_003',
            'task_subtype': 'resolve',
            'completion_threshold': 0.75,
            'max_steps': 6,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['resolve_conflict']},
            'conflict_packages': ['torch', 'transformers'],
            'compatibility_matrix': {
                'torch': {
                    '2.1.0': {'transformers': '>=4.35'},
                    '2.0.0': {'transformers': '>=4.30'},
                },
                'transformers': {
                    '4.37.0': {'torch': '>=2.0'},
                    '4.35.0': {'torch': '>=2.0'},
                    '4.30.0': {'torch': '>=1.13'},
                },
            },
            'requirements': {'torch': '1.11.0', 'transformers': '4.20.0'},
            'code_snippet': '''# requirements.txt  
torch==1.11.0
transformers==4.20.0''',
            'task_description': 'Resolve conflict between PyTorch and Transformers library versions.',
        },
    ],
    'dep_hard': [
        {
            'case_id': 'dep_hard_001',
            'task_subtype': 'migrate',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 2, 'required_sequence': ['migrate_api']},
            'graph_breaks': ['break_001', 'break_002', 'break_003'],
            'checklist_dependency_graph': {
                'break_003': ['break_001', 'break_002'],
                'break_002': ['break_001'],
                'break_001': [],
            },
            'correct_fix_map': {
                'break_001': 'torch.where',
                'break_002': 'tensor.shape[0]',
                'break_003': '.detach().numpy()',
            },
            'code_snippet': '''import torch

@torch.compile
def forward(x):
    # break_001: data-dependent control flow
    if x.item() > 0.5:
        x = x * 2
    
    # break_002: Python builtin on tensor
    batch_size = len(x)
    
    # break_003: numpy conversion inside compile
    result = x.numpy()
    return result''',
            'break_descriptions': [
                'break_001: line 6 — data-dependent control flow: if x.item() > 0.5',
                'break_002: line 9 — Python builtin on tensor: len(x)',
                'break_003: line 12 — numpy inside compiled function: x.numpy()',
            ],
            'graph_break_report': [
                'break_001: line 6 — data-dependent control flow: if x.item() > 0.5',
                'break_002: line 9 — Python builtin on tensor: len(x)',
                'break_003: line 12 — numpy inside compiled function: x.numpy()',
            ],
            'task_description': 'This PyTorch model uses torch.compile but has multiple graph-break patterns. Fix them in dependency order.',
        },
        {
            'case_id': 'dep_hard_002',
            'task_subtype': 'migrate',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 2, 'required_sequence': ['migrate_api']},
            'graph_breaks': ['break_a', 'break_b', 'break_c', 'break_d'],
            'checklist_dependency_graph': {
                'break_d': ['break_b', 'break_c'],
                'break_c': ['break_a'],
                'break_b': ['break_a'],
                'break_a': [],
            },
            'correct_fix_map': {
                'break_a': 'torch.where',
                'break_b': 'tensor.shape[0]',
                'break_c': 'torch.tensor',
                'break_d': '.detach()',
            },
            'code_snippet': '''import torch

@torch.compile(fullgraph=True)
def training_step(model, x, labels):
    # break_a: data-dependent branch
    if x.max().item() > 1.0:
        x = x / x.max()
    
    # break_b: Python len() on tensor
    n_samples = len(x)
    
    # break_c: Python list to tensor inside compile
    weights = torch.FloatTensor([1.0, 2.0, 3.0])
    
    # break_d: in-place operation on leaf tensor
    x += 0.1  # in-place modification
    
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, labels)
    return loss''',
            'break_descriptions': [
                'break_a: line 6 — data-dependent: if x.max().item() > 1.0',
                'break_b: line 10 — Python builtin: len(x)',
                'break_c: line 13 — legacy constructor: torch.FloatTensor()',
                'break_d: line 16 — in-place op on leaf: x += 0.1',
            ],
            'graph_break_report': [
                'break_a: line 6 — data-dependent: if x.max().item() > 1.0',
                'break_b: line 10 — Python builtin: len(x)',
                'break_c: line 13 — legacy constructor: torch.FloatTensor()',
                'break_d: line 16 — in-place op on leaf: x += 0.1',
            ],
            'task_description': 'Fix all 4 graph-break patterns in this compiled training step. Dependencies must be resolved in order.',
        },
    ],
}
