import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + 3*x*y + y**2 - 4*x - 5*y
grad_f = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
H = sp.hessian(f, (x, y))
H_inv = H.inv()

product = H_inv * grad_f

print("1) Gradient ∇f(x,y):\n", grad_f)
print("\n2) Hessian H(x,y):\n", H)
print("   Inverse Hessian H(x,y)^(-1):\n", H_inv)
print("\n3) Product H(x,y)^(-1) ∇f(x,y):\n", sp.simplify(product))