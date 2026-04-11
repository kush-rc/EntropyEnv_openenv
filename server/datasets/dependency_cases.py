# server/datasets/dependency_cases.py
# Ground truth cases for PyTorch Migration Time-Machine tasks.
#
# CRITICAL FIX:
# dep_hard previously had:
#   done_conditions: {min_actions: 2, required_sequence: ['migrate_api', 'migrate_api']}
#
# This caused TWO bugs:
#   1. The agent called migrate_api once. Router checked Counter: needs 2, has 1 → not done.
#   2. Agent called migrate_api again → repetition_penalty fires (-0.20), tanking the score.
#   3. Episode only ends at max_steps with a broken accumulated score.
#
# FIX: dep_hard now uses min_actions=1, required_sequence=['migrate_api'].
# The task is already hard enough from the grader — complex checklist, ordering
# constraints, and exact token matching in fix_quality. The done condition
# should not add extra difficulty on top of this.
#
# ALL dep_easy, dep_medium, dep_hard done conditions verified below.

DEPENDENCY_CASES = {

    # ── DEP EASY ─────────────────────────────────────────────────────────
    # Task: flag outdated packages and deprecated API usage.
    # Done: after 1 flag_outdated action.
    # Grader: F1 on packages (precision+recall) × 0.55 + deprecated_api_match × 0.45
    # ─────────────────────────────────────────────────────────────────────
    'dep_easy': [
        {
            'case_id': 'dep_easy_001',
            'task_subtype': 'flag',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'task_description': (
                'This codebase uses torch==1.9.0 and relies on torch.autograd.Variable. '
                'Flag all outdated packages and the deprecated API.'
            ),
            'code_snippet': (
                'import torch\n'
                'from torch.autograd import Variable\n'
                'x = Variable(torch.randn(3, 4))\n'
                'model = torch.nn.Linear(4, 2)\n'
                'out = model(x)'
            ),
            'requirements': {'torch': '1.9.0', 'torchvision': '0.10.0'},
            'expected_outdated_packages': ['torch', 'torchvision'],
            'expected_deprecated_api': 'torch.autograd.Variable',
            'expected_replacement': 'plain tensor with requires_grad=True',
        },
        {
            'case_id': 'dep_easy_002',
            'task_subtype': 'flag',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'task_description': (
                'This codebase uses torch==1.4.0 and calls .cuda() directly. '
                'Flag outdated packages and the deprecated device assignment pattern.'
            ),
            'code_snippet': (
                'import torch\n'
                'model = MyModel()\n'
                'model.cuda()  # deprecated — use .to(device)\n'
                'tensor = torch.randn(2, 3).cuda()'
            ),
            'requirements': {'torch': '1.4.0'},
            'expected_outdated_packages': ['torch'],
            'expected_deprecated_api': '.cuda()',
            'expected_replacement': '.to(device)',
        },
        {
            'case_id': 'dep_easy_003',
            'task_subtype': 'flag',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'task_description': (
                'This codebase uses torch==1.7.0 with DataParallel. '
                'Flag the outdated package and the deprecated multi-GPU API.'
            ),
            'code_snippet': (
                'import torch\n'
                'model = torch.nn.DataParallel(MyModel())\n'
                'model.cuda()'
            ),
            'requirements': {'torch': '1.7.0', 'numpy': '1.18.0'},
            'expected_outdated_packages': ['torch', 'numpy'],
            'expected_deprecated_api': 'torch.nn.DataParallel',
            'expected_replacement': 'DistributedDataParallel',
        },
        {
            'case_id': 'dep_easy_004',
            'task_subtype': 'flag',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'task_description': (
                'Flag outdated packages and the deprecated ONNX export API in this code.'
            ),
            'code_snippet': (
                'import torch\n'
                'torch.onnx.export(model, dummy_input, "model.onnx",\n'
                '                  opset_version=9,\n'
                '                  enable_onnx_checker=True)  # deprecated kwarg'
            ),
            'requirements': {'torch': '1.8.0'},
            'expected_outdated_packages': ['torch'],
            'expected_deprecated_api': 'enable_onnx_checker',
            'expected_replacement': 'remove the kwarg (deprecated in 1.9, removed in 2.0)',
        },
        {
            'case_id': 'dep_easy_005',
            'task_subtype': 'flag',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['flag_outdated']},
            'task_description': (
                'Flag outdated packages and the deprecated autocast API.'
            ),
            'code_snippet': (
                'import torch\n'
                'from torch.cuda.amp import autocast\n'
                'with autocast():  # deprecated import path\n'
                '    output = model(input)'
            ),
            'requirements': {'torch': '1.6.0', 'torchaudio': '0.6.0'},
            'expected_outdated_packages': ['torch', 'torchaudio'],
            'expected_deprecated_api': 'torch.cuda.amp.autocast',
            'expected_replacement': 'torch.amp.autocast',
        },
    ],

    # ── DEP MEDIUM ────────────────────────────────────────────────────────
    # Task: resolve version conflicts using the compatibility_matrix.
    # Done: after 1 resolve_conflict action.
    # Grader: valid_pkgs/conflict_count + cross-constraint check - downgrade penalty
    # ─────────────────────────────────────────────────────────────────────
    'dep_medium': [
        {
            'case_id': 'dep_medium_001',
            'task_subtype': 'resolve',
            'completion_threshold': 0.70,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['resolve_conflict']},
            'task_description': (
                'Resolve the version conflict between torch, numpy, and protobuf. '
                'Use the compatibility_matrix to find a compatible set of versions.'
            ),
            'code_snippet': 'requirements.txt with conflicting torch==2.0.0, numpy==1.20.0, protobuf==3.9.0',
            'requirements': {'torch': '2.0.0', 'numpy': '1.20.0', 'protobuf': '3.9.0'},
            'conflict_packages': ['torch', 'numpy', 'protobuf'],
            'compatibility_matrix': {
                'torch': {
                    '2.1.0': {'numpy': '>=1.21,<2.0', 'protobuf': '>=3.20,<5.0'},
                    '2.0.0': {'numpy': '>=1.20,<1.25', 'protobuf': '>=3.19,<4.0'},
                },
                'numpy': {
                    '1.24.0': {},
                    '1.21.0': {},
                    '1.20.0': {},
                },
                'protobuf': {
                    '4.23.0': {},
                    '3.20.0': {},
                    '3.9.0': {'torch': '<=1.13'},
                },
            },
        },
        {
            'case_id': 'dep_medium_002',
            'task_subtype': 'resolve',
            'completion_threshold': 0.70,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['resolve_conflict']},
            'task_description': (
                'Resolve the version conflict between tensorflow, keras, and h5py.'
            ),
            'code_snippet': 'requirements.txt: tensorflow==2.10.0, keras==2.10.0, h5py==2.10.0',
            'requirements': {'tensorflow': '2.10.0', 'keras': '2.10.0', 'h5py': '2.10.0'},
            'conflict_packages': ['tensorflow', 'keras', 'h5py'],
            'compatibility_matrix': {
                'tensorflow': {
                    '2.13.0': {'keras': '>=2.13,<2.14', 'h5py': '>=3.7'},
                    '2.10.0': {'keras': '==2.10.0', 'h5py': '>=3.1'},
                },
                'keras': {
                    '2.13.0': {'tensorflow': '>=2.13,<2.14'},
                    '2.10.0': {'tensorflow': '==2.10.0'},
                },
                'h5py': {
                    '3.9.0': {},
                    '3.7.0': {},
                    '2.10.0': {'tensorflow': '<=2.3'},
                },
            },
        },
        {
            'case_id': 'dep_medium_003',
            'task_subtype': 'resolve',
            'completion_threshold': 0.70,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['resolve_conflict']},
            'task_description': (
                'Resolve the conflict between transformers, tokenizers, and datasets packages.'
            ),
            'code_snippet': 'requirements: transformers==4.20.0, tokenizers==0.11.0, datasets==1.18.0',
            'requirements': {'transformers': '4.20.0', 'tokenizers': '0.11.0', 'datasets': '1.18.0'},
            'conflict_packages': ['transformers', 'tokenizers', 'datasets'],
            'compatibility_matrix': {
                'transformers': {
                    '4.35.0': {'tokenizers': '>=0.14,<0.19', 'datasets': '>=2.14'},
                    '4.20.0': {'tokenizers': '>=0.11,<0.14', 'datasets': '>=1.18'},
                },
                'tokenizers': {
                    '0.15.0': {'transformers': '>=4.28'},
                    '0.14.0': {'transformers': '>=4.25'},
                    '0.11.0': {},
                },
                'datasets': {
                    '2.14.0': {},
                    '2.10.0': {},
                    '1.18.0': {'tokenizers': '<=0.13'},
                },
            },
        },
    ],

    # ── DEP HARD ──────────────────────────────────────────────────────────
    # Task: fix torch.compile graph-break patterns.
    # Done: after 1 migrate_api action (FIXED from 2 → 1).
    #
    # IMPORTANT: min_actions=1, required_sequence=['migrate_api']
    # The grader already makes this hard through:
    #   - Multiple graph_breaks to fix (3-5 per case)
    #   - Ordering constraints via checklist_dependency_graph
    #   - Exact token matching in fix_quality
    # We do NOT need the done condition to create artificial difficulty.
    # ─────────────────────────────────────────────────────────────────────
    'dep_hard': [
        {
            'case_id': 'dep_hard_001',
            'task_subtype': 'migrate',
            'completion_threshold': 0.60,
            'max_steps': 6,
            # FIXED: was min_actions=2, required_sequence=['migrate_api','migrate_api']
            # which caused repetition penalty on the 2nd call and never terminated cleanly
            'done_conditions': {'min_actions': 1, 'required_sequence': ['migrate_api']},
            'task_description': (
                'Fix the torch.compile graph-break patterns in this training loop. '
                'Provide completed_items (list of break IDs) and code_changes (dict of fixes).'
            ),
            'code_snippet': (
                'import torch\n\n'
                'def train_step(model, x):\n'
                '    out = model(x)\n'
                '    if out.shape[0] != x.shape[0]:   # data-dependent branch [break_001]\n'
                '        out = torch.zeros_like(x)\n'
                '    idx = int(out.argmax())           # int() conversion [break_002]\n'
                '    mask = out > 0.5                  # dynamic masking [break_003]\n'
                '    return out[mask].sum()\n'
            ),
            'graph_break_report': [
                'break_001: data-dependent control flow (if out.shape[0] != x.shape[0])',
                'break_002: Python int() call on tensor (int(out.argmax()))',
                'break_003: dynamic boolean indexing (out[mask])',
            ],
            'graph_breaks': ['break_001', 'break_002', 'break_003'],
            'checklist_dependency_graph': {
                'break_003': ['break_002'],  # must fix int() conversion before mask
            },
            'correct_fix_map': {
                'break_001': 'torch.where',
                'break_002': 'torch.argmax',
                'break_003': 'torch.masked_select',
            },
        },
        {
            'case_id': 'dep_hard_002',
            'task_subtype': 'migrate',
            'completion_threshold': 0.60,
            'max_steps': 6,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['migrate_api']},
            'task_description': (
                'Fix these torch.compile graph-breaks in a model forward pass.'
            ),
            'code_snippet': (
                'def forward(self, x):\n'
                '    x = self.conv(x)\n'
                '    size = x.size(0)           # .size() with int [break_001]\n'
                '    out = x.numpy()            # .numpy() call [break_002]\n'
                '    out = torch.from_numpy(out)\n'
                '    return out[:size//2]       # dynamic slice [break_003]\n'
            ),
            'graph_break_report': [
                'break_001: .size() call returning Python int',
                'break_002: .numpy() call breaks compilation boundary',
                'break_003: dynamic slicing with Python division',
            ],
            'graph_breaks': ['break_001', 'break_002', 'break_003'],
            'checklist_dependency_graph': {
                'break_003': ['break_001'],
            },
            'correct_fix_map': {
                'break_001': 'tensor.shape[0]',
                'break_002': 'detach',
                'break_003': 'torch.narrow',
            },
        },
        {
            'case_id': 'dep_hard_003',
            'task_subtype': 'migrate',
            'completion_threshold': 0.60,
            'max_steps': 6,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['migrate_api']},
            'task_description': (
                'Fix torch.compile graph-breaks in this attention implementation.'
            ),
            'code_snippet': (
                'def attention(q, k, v):\n'
                '    scores = torch.matmul(q, k.transpose(-2, -1))\n'
                '    if scores.max() > 100:      # data-dependent branch [break_001]\n'
                '        scores = scores / 100\n'
                '    weights = scores.numpy()    # numpy call [break_002]\n'
                '    weights = torch.softmax(torch.tensor(weights), dim=-1)\n'
                '    n = int(q.shape[0])         # Python int [break_003]\n'
                '    return weights[:n] @ v\n'
            ),
            'graph_break_report': [
                'break_001: data-dependent branch on scores.max()',
                'break_002: .numpy() breaks torch.compile boundary',
                'break_003: Python int() on tensor dimension',
            ],
            'graph_breaks': ['break_001', 'break_002', 'break_003'],
            'checklist_dependency_graph': {
                'break_003': ['break_001'],
                'break_002': ['break_001'],
            },
            'correct_fix_map': {
                'break_001': 'torch.clamp',
                'break_002': 'torch.softmax',
                'break_003': 'tensor.shape',
            },
        },
        {
            'case_id': 'dep_hard_004',
            'task_subtype': 'migrate',
            'completion_threshold': 0.60,
            'max_steps': 6,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['migrate_api']},
            'task_description': (
                'Fix four torch.compile graph-breaks in this training utility.'
            ),
            'code_snippet': (
                'def process_batch(batch):\n'
                '    lengths = [len(x) for x in batch]   # Python list comp [break_001]\n'
                '    max_len = max(lengths)               # Python max() [break_002]\n'
                '    padded  = torch.zeros(len(batch), max_len)\n'
                '    for i, x in enumerate(batch):        # Python loop [break_003]\n'
                '        padded[i, :len(x)] = x\n'
                '    out = model(padded)\n'
                '    return out.cpu().numpy()             # .numpy() [break_004]\n'
            ),
            'graph_break_report': [
                'break_001: Python list comprehension over tensor data',
                'break_002: Python max() on list of tensor values',
                'break_003: Python for loop with tensor indexing',
                'break_004: .numpy() call at output',
            ],
            'graph_breaks': ['break_001', 'break_002', 'break_003', 'break_004'],
            'checklist_dependency_graph': {
                'break_002': ['break_001'],
                'break_003': ['break_002'],
            },
            'correct_fix_map': {
                'break_001': 'torch.tensor',
                'break_002': 'torch.max',
                'break_003': 'torch.nn.utils.rnn.pad_sequence',
                'break_004': 'detach',
            },
        },
        {
            'case_id': 'dep_hard_005',
            'task_subtype': 'migrate',
            'completion_threshold': 0.60,
            'max_steps': 6,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['migrate_api']},
            'task_description': (
                'Fix torch.compile graph-breaks caused by vmap incompatibilities.'
            ),
            'code_snippet': (
                'from torch._vmap_internals import vmap  # deprecated [break_001]\n'
                'import functorch                         # deprecated module [break_002]\n\n'
                'def batched_fn(x):\n'
                '    result = vmap(model)(x)\n'
                '    if result.isnan().any():             # data-dependent check [break_003]\n'
                '        result = torch.zeros_like(result)\n'
                '    return result\n'
            ),
            'graph_break_report': [
                'break_001: torch._vmap_internals.vmap is deprecated (use torch.vmap)',
                'break_002: functorch module is deprecated (merged into torch)',
                'break_003: data-dependent .any() check breaks compilation',
            ],
            'graph_breaks': ['break_001', 'break_002', 'break_003'],
            'checklist_dependency_graph': {
                'break_002': ['break_001'],
            },
            'correct_fix_map': {
                'break_001': 'torch.vmap',
                'break_002': 'torch.func',
                'break_003': 'torch.where',
            },
        },
    ],
}
