import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def calc_performance_metric(self):
    """Calculate hypervolume to the reference point"""
    front = self.pareto_fronts[0]
    solutions = np.array([solution.get_fitness(is_flag=False) for solution in self.population[front]])
    self.perf_metrics.append(
        [self.n_iters, self.ind(solutions)]
    )

def visualise_convergence_plot(filename, f, color, label, title, figsize, xlabel, ylabel):
    N_ITERATIONS = len(f)
    lower_bound = [] * N_ITERATIONS
    upper_bound = [] * N_ITERATIONS
    median = [] * N_ITERATIONS

    for i in range(N_ITERATIONS):
        lower_bound.append(np.min(f[i]))
        upper_bound.append(np.max(f[i]))
        median.append(np.median(f[i]))

    plt.figure(figsize=figsize)

    plt.plot(
        range(N_ITERATIONS),
        lower_bound,
        color=color,
        alpha=.3,
    )
    plt.plot(
        range(N_ITERATIONS),
        upper_bound,
        color=color,
        alpha=.3,
    )
    plt.plot(
        range(N_ITERATIONS),
        median,
        label=label,
        c=color
    )
    plt.fill_between(range(N_ITERATIONS), lower_bound, upper_bound, color=color, alpha=0.2)

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def visualize_box_plot(filename, fs, labels, figsize): 
    plt.figure(figsize=figsize)

    data = []
    bp_labels = []

    for j in range(2):
        for i in range(len(fs)):
            f = fs[i]
            norm_f = f[:, j] / f[:, j].max()
            data.append(norm_f)
            bp_labels.append(f'{labels[i]}\n{"energy\nconsumption" if j == 0 else "load\nbalance"} ')

    plt.boxplot(
        data,
        labels=bp_labels,
    )

    plt.savefig(filename)
    plt.close()

def visualize_objective_space(filename, list_PFs, fs, labels, figsize, is_non_dominated):
    FIRST_INDEX = 0
    plt.figure(figsize=figsize)

    for i in range(len(fs)):
        PFs = list_PFs[i]
        f = fs[i]

        front = PFs[FIRST_INDEX]
        non_dominated = f[front]

        if i == 1:
            plt.scatter(
                x=f[:, 0],
                y=f[:, 1],
                label=f'{labels[i]} solution',
            )
            continue

        if is_non_dominated[i]:
            dominated = np.array([])
            for j in range(1, len(PFs)):
                if len(dominated) == 0:
                    dominated = f[PFs[j]]
                dominated = np.append(dominated, f[PFs[j]], axis=0)
            dominated = np.array(dominated)

            if dominated.size != 0:
                plt.scatter(
                    x=dominated[:, 0],
                    y=dominated[:, 1],
                    label=f'{labels[i]} dominated solution',
                )

        plt.scatter(
            x=non_dominated[:, 0],
            y=non_dominated[:, 1],
            label=f'{labels[i]} non-dominated solution',
        )

    plt.xlabel('Energy Consumption')
    plt.ylabel('Load Balance')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def generate_video_from_images(image_folder, output_video_path, frame_rate=1):
    # Get list of all image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Ensure images are in correct order if they are named sequentially

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write out frame to video

    # Release everything if job is finished
    video.release()
    cv2.destroyAllWindows()
