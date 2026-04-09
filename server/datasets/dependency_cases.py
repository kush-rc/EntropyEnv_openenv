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
        {
            'case_id': 'dep_hard_003',
            'task_subtype': 'migrate',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 2, 'required_sequence': ['migrate_api']},
            'graph_breaks': ['break_x', 'break_y', 'break_z'],
            'checklist_dependency_graph': {
                'break_z': ['break_x'],  # z depends on x
                'break_y': [],           # y is independent
                'break_x': [],           # x is independent
            },
            'correct_fix_map': {
                'break_x': 'tensor.numel()',
                'break_y': 'torch.jit.script',
                'break_z': 'torch.no_grad()',
            },
            'code_snippet': '''import torch

@torch.compile
def forward(x, mask):
    # break_x: tensor.size() returns Python int (graph break)
    n = x.size(0) * x.size(1)
    
    # break_y: Python function call inside compile
    def custom_fn(t):
        return t * 2
    x = custom_fn(x)
    
    # break_z: gradient tracking inside compiled region
    with torch.enable_grad():  # breaks graph
        x = x * mask
    
    return x''',
            'break_descriptions': [
                'break_x: line 6 — tensor.size() returns Python int, use tensor.numel() instead',
                'break_y: line 10 — Python function call, use torch.jit.script decorator',
                'break_z: line 14 — enable_grad inside compile, use torch.no_grad() for inference',
            ],
            'graph_break_report': [
                'break_x: line 6 — tensor.size() returns Python int, use tensor.numel() instead',
                'break_y: line 10 — Python function call, use torch.jit.script decorator',
                'break_z: line 14 — enable_grad inside compile, use torch.no_grad() for inference',
            ],
            'task_description': 'Fix torch.compile graph breaks in this custom layer. Note dependency: break_z needs break_x fixed first.',
        },
        {
            'case_id': 'dep_hard_004',
            'task_subtype': 'migrate',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 2, 'required_sequence': ['migrate_api']},
            'graph_breaks': ['break_alpha', 'break_beta', 'break_gamma', 'break_delta'],
            'checklist_dependency_graph': {
                'break_delta': ['break_beta', 'break_gamma'],  # delta needs both
                'break_gamma': ['break_alpha'],                # gamma needs alpha
                'break_beta': [],
                'break_alpha': [],
            },
            'correct_fix_map': {
                'break_alpha': 'torch.where',
                'break_beta': 'tensor.shape[0]',
                'break_gamma': 'torch.stack',
                'break_delta': '@torch.jit.script',
            },
            'code_snippet': '''import torch

@torch.compile(fullgraph=True)
def loss_fn(pred, target, weights):
    # break_alpha: if statement on tensor value
    if target.sum() > 0:
        pred = pred * 1.5
    
    # break_beta: len() on tensor
    batch_size = len(pred)
    
    # break_gamma: Python list → tensor conversion
    normalized = []
    for i in range(batch_size):
        normalized.append(pred[i] / weights[i])
    result = torch.tensor(normalized)  # breaks graph
    
    # break_delta: calls non-scripted helper
    def helper(x):
        return x.clamp(0, 1)
    return helper(result)''',
            'break_descriptions': [
                'break_alpha: line 6 — data-dependent control flow, use torch.where(condition, ...)',
                'break_beta: line 10 — len() builtin on tensor, use tensor.shape[0]',
                'break_gamma: line 16 — torch.tensor() on Python list, use torch.stack()',
                'break_delta: line 20 — unscripted helper function, add @torch.jit.script decorator',
            ],
            'graph_break_report': [
                'break_alpha: line 6 — data-dependent control flow, use torch.where(condition, ...)',
                'break_beta: line 10 — len() builtin on tensor, use tensor.shape[0]',
                'break_gamma: line 16 — torch.tensor() on Python list, use torch.stack()',
                'break_delta: line 20 — unscripted helper function, add @torch.jit.script decorator',
            ],
            'task_description': 'Complex graph-break cascade. Delta depends on Beta AND Gamma. Gamma depends on Alpha. Fix in dependency order.',
        },
        {
            'case_id': 'dep_hard_005',
            'task_subtype': 'migrate',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 2, 'required_sequence': ['migrate_api']},
            'graph_breaks': ['break_001', 'break_002', 'break_003'],
            'checklist_dependency_graph': {
                'break_003': ['break_001', 'break_002'],
                'break_002': [],
                'break_001': [],
            },
            'correct_fix_map': {
                'break_001': 'torch.compile(disable=True)',
                'break_002': 'functorch.vmap',
                'break_003': 'torch.export',
            },
            'code_snippet': '''import torch
from torch.nn.utils import clip_grad_norm_

@torch.compile
def training_step(model, batch, optimizer):
    # break_001: optimizer.step() inside compiled region
    loss = model(batch['x'], batch['y'])
    loss.backward()
    optimizer.step()  # graph break
    
    # break_002: Python loop over batch dimension
    grads = []
    for param in model.parameters():
        grads.append(param.grad.norm())
    
    # break_003: clip_grad_norm_ mutation
    clip_grad_norm_(model.parameters(), max_norm=1.0)  # breaks graph
    
    return loss.item()''',
            'break_descriptions': [
                'break_001: line 9 — optimizer.step() not compilable, wrap optimizer logic outside compile',
                'break_002: line 13 — Python loop batching, use functorch.vmap for vectorization',
                'break_003: line 17 — in-place grad clipping, use torch.export with explicit mutation tracking',
            ],
            'graph_break_report': [
                'break_001: line 9 — optimizer.step() not compilable, wrap optimizer logic outside compile',
                'break_002: line 13 — Python loop batching, use functorch.vmap for vectorization',
                'break_003: line 17 — in-place grad clipping, use torch.export with explicit mutation tracking',
            ],
            'task_description': 'Fix training loop graph breaks. Optimizer, gradient accumulation, and clipping all cause compilation failures.',
        },
    ],
}
