import matplotlib.pyplot as plt
wr1,wr2=[],[]
with open("winrate.txt", "r") as f:
    for line in f:
        line_data = line.strip().split(',')
        wr1.append(float(line_data[0]))
        wr2.append(float(line_data[1]))
plt.figure(figsize=(12,8))
plt.plot(range(100),wr1,label = "ucb1", color ='b')
plt.plot(range(100),wr2,label = "ucb1 tuned", color = 'r')
plt.legend(loc='best')
plt.title('Winrate over time')  
plt.xlabel('Games')
plt.ylabel('Winrate')
plt.savefig("Winrate_mcts.png")
