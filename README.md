# UAV-SYSTEM
Multi UAV Path Planning using custom MARLLIB

실행방법

1. UAV_system.zip의 압축을 푼다  
2. 압축해제한 경우(UAV_system)의 marllib\envs\base_env\config\multi_agent\mlagent.yaml에서 "env_path"의 값을 UAV_system 폴더에 있는 env 폴더의 절대 경로로 설정해준다. (예시: "C:\Users\username\Desktop\custom_MARLlib\env")  
3. Hyperparameter 설정:  
   - marllib\marl\algos\w\hyperparams\common\ippo.yaml에서 다양한 하이퍼파라미터를 수정할 수 있다.  
   - marllib\envs\base_env\multi_agent_mlagent.py의 128번째 줄(#Customize model)에 있는 모델 구조를 수정할 수 있으며 139번째 줄에서 모델을 학습하는 전체 step 수를 조정할 수 있다.  
4. UAV_system 디렉터리 상에서 터미널에 python marllib/envs/base_env/multi_agent_mlagent.py를 입력하면 학습이 시작된다. 학습 중 결과는 터미널의 tensorboard --logdir './'를 입력하면 확인 가능하며 학습 완료 후 exp_results\ippo_mlp_multi_agent_mlagent 폴더에 있는 각 실험 폴더에서 세부 결과를 확인가능하다.  
5. 학습 완료한 모델의 테스트 결과를 확인하고 싶은 경우:  
   hyperparams\common\ippo.yaml에서 "lr"의 값을 0으로 설정한 뒤, marllib\envs\base_env\multi_agent_mlagent.py에서 36번째 줄에 있는 time_scale=12.0에서 1.0으로 변경한다. 또한 145~149번째 줄의 주석을 해제한 뒤, 실행결과 저장된 폴더에서 실행결과 확인을 원하는 checkpoint 번호를 선택하여 다음과 같이 작성해준다.  

예시:  
restore_path=  
{  
   'params_path': r'(실험 결과 저장된 폴더 경로)\params.json',  
   'model_path': r'(실험 결과 저장된 폴더 경로)\checkpoint_000070',  
   'render': True  
}  

그 다음 UAV_system 디렉터리 상에서 터미널에 python marllib/envs/base_env/multi_agent_mlagent.py를 입력하면 테스트 결과를 확인할 수 있다.
