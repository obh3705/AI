import torch

class TwoLayerNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		"""
		생성자에서 2개의 nn.Linear 모듈을 생성하고 멤버 변수로 지정
		"""
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)

	def forward(self, x):
		"""
		순전파 함수에서는 입력 데이터의 Tensor를 받고 출력 데이터의 Tensor를
		반환해야 한다. Tensor 상의 임의읭 연산자뿐만 아니라 생성자에서 정의한
		Module도 사용할 수 있다.
		"""
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred

# N은 배치 크기, D_in은 입력의 차원
# H는 은닉층의 차원, D_out은 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 앞에서 정의한 클래스를 생성하여 모델을 구성한다.
model = TwoLayerNet(D_in, H, D_out)

# 손실 함수와 Optimizer를 만든다. SGD 생성자에 model.parameters()를 호출하면
# 모델의 멤버인 2개의 nn.Linear 모듈의 학습 가능한 매개변수들이 포함된다.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
	# 순전파 단계: 모델에 x를 전달하여 예상되는 y 값을 계산
	y_pred = model(x)

	# 손실을 계산하고 출력
	loss = criterion(y_pred, y)
	if t % 100 == 99:
		print(t, loss.item())

	# 변화도를 0으로 만들고 역전파 단계를 수행하고 가중치를 갱신한다.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()