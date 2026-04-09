import sys
sys.path.append('.')
from inference import parse_action, MAX_STEPS
import functools

# Mock the print function so we can capture exactly what the user wants to see
def run_mock():
    task_id = "sec_easy"
    episode_id = "ep-abc12345"
    step_num = 1
    display_action = "identify_vulnerability"
    reward = 0.5
    done = False
    
    print(f'[START] task_id={task_id} episode_id={episode_id}')
    print(f'[STEP] task_id={task_id} step={step_num} action={display_action} reward={reward:.4f} done={done}')
    
    step_num = 2
    display_action = "propose_fix"
    reward = 1.0
    done = True
    print(f'[STEP] task_id={task_id} step={step_num} action={display_action} reward={reward:.4f} done={done}')
    
    total_reward = 1.5
    print(f'[END] task_id={task_id} episode_id={episode_id} total_reward={total_reward:.4f} steps={step_num}')

run_mock()
