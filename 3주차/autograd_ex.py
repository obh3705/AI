import torch

dtype = torch.float
device = torch.device("cpu")

# N은 배치 크기이며, D_in은 입력 단계의 차원
# H는 은닉층의 차원, D_out은 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성
# requires_grad=False로 설정하여 역전파 중에서 이 Tensor들에 대한 변화도를 계산할
# 필요가 없음을 나타낸다. (requres_grad의 기본값이 False 코드에서 반영하지 않음)
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성
# requires_grad=True로 설정하여 역전파 중에 이 Tensor들에 대한 변화도를 계산할 필요가 있음
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
	# 순전파 단계: Tensor 연산을 사용하여 예상되는 y값을 계산.
	# 이는 Tensor를 사용한 순전파 단계와 완전히 동일하지만, 역전파 단계를 별도로 구현하지
	# 않아도 되므로 중간값들에 대한 참조를 갖고 있을 필요가 없음
	y_pred = x.mm(w1).clamp(min=0).mm(w2)

	# Tensor 연산을 사용하여 손실을 계산하고 출력
	# loss는 (1,) 형태의 Tensor, loss.item()은 loss의 스칼라 값
	loss = (y_pred - y).pow(2).sum()
	if t % 100 == 99:
		print(t, loss.item())

	# autograd를 사용하여 역전파 단계를 계산. 이는 requires_grad=True를 갖는 
	# 모든 Tensor에 대해 손실의 변화도를 계산. 이후 w1.grad와 w2.grad는
	# w1과 w2 각각에 대한 손실의 변화도를 갖는 Tensor가 된다.
	loss.backward()

	# 경사하강법을 사용하여 가중치를 수동으로 갱신
	# torch.no_grad()로 감싸는 이유는 가중치들이 requires_grad=True 이지만
	# autograd에서는 이를 추적할 필요가 없다.
	# 다른 방법은 weight.data 및 weight.grad.data를 조작하는 방법이 있다.
	# tensor.data가 tensor의 저장공간을 공유하긴 하지만 이력을 추적하지 않는다.
	# 이를 위해 torch.optim.SGD를 사용할 수 있다.
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad

		# 가중치 갱신 후에는 수동으로 변화도를 0으로 만든다.
		w1.grad.zero_()
		w2.grad.zero_()