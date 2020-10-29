import torch

# N은 배치 크기, D_in은 입력의 차원
# H는 은닉층의 차원, D_out은 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# nn 패키지를 사용하여 모델과 손실 함수를 정의
model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H),
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# optim 패키지를 사용하여 모델의 가중치를 갱신할 Optimizer를 정의한다.
# Adam 생성자의 첫번째 인자는 어떤 Tensor가 갱신되어야 하는지 알려준다.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
	# 순전파 단계: 모델에 x를 전달하여 예상되는 y 값을 계산
	y_pred = model(x)

	# 손실을 계산하고 출력
	loss = loss_fn(y_pred, y)
	if t % 100 == 99:
		print(t, loss.item())

	# 역전파 단계 전에 Optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인)
	# 갱신할 변수들에 대한 모든 변화도를 0으로 만든다.
	# 기본적으로 .backward()를 호출할 때마다 변화도가 버퍼에 누적되기 때문이다.
	optimizer.zero_grad()

	# 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산
	loss.backward()

	# Optimizer의 step 함수를 호출하면 매개변수가 갱신된다.
	optimizer.step()