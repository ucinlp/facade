import matplotlib.pyplot as plt 

def main():
  with open("simple_grad_pmi_prem_corr.txt", "r") as f:
    x = []
    y = []
    for idx, line in enumerate(f.readlines()):
      rho = float(line.split(' ')[-1])
      y.append(rho)
      x.append(idx)
    
    plt.plot(x, y, 'bo', markersize=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Spearman Rank Coefficient")
    plt.savefig("simple_grad_pmi_prem_corr.png")

if __name__ == "__main__":
  main()