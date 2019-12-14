from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import re
import os

def check(env,agent,save_mp4=None):
    save_mp4= save_mp4 or 'result.mp4'        
    frames = []
    for i in range(3):
        obs = env.reset()
        done = False
        R = 0
        t = 0
        while not done and t < 200:
            frames.append(env.render(mode = 'rgb_array'))
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
        print('test episode:', i, 'R:', R)
        agent.stop_episode()
    env.close()

    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off') 
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),interval=50)
    anim.save(save_mp4)
    return anim

def growth(env,agent,outdir,max_num=200,save_mp4=None):
    save_mp4= save_mp4 or 'result.mp4'        

    files = os.listdir(outdir)
    files_dir = [f for f in files if os.path.isdir(os.path.join(outdir, f))]
    agentList=[]
    for f in files_dir:
        if re.search('_',f):
            f2=f.split('_')
            agentList.append([int(f2[0]),f])
    agentList.sort()

    frames = []
    for item in agentList:
        agent.load(outdir+'/'+item[1])
        obs = env.reset()
        done = False
        R = 0
        t = 0
        while not done and t < max_num:
            frames.append(env.render(mode = 'rgb_array'))
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
        print('agent:', item[1], 'R:', R)
        agent.stop_episode()
    env.close()
    from IPython.display import HTML
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off') 
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),interval=50)
    anim.save(save_mp4)
    return anim

