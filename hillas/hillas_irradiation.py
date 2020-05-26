import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import scipy.misc
from scipy.misc import imread
import matplotlib
matplotlib.use('Agg')

# Calculate Hillas parameters for irradiation images

# from Brent, calculate Hillas parameters of events
#charge_coords = [[x_coords], [y_coords], [charges]]


def hillas(charge_coords):
    # print(charge_coords.shape)
    x = 0
    y = 0
    x2 = 0
    y2 = 0
    xy = 0
    CHARGE = 0
    # print(charge_coords.shape)
    CHARGE = np.sum(charge_coords[2])
    x = np.sum(charge_coords[0] * charge_coords[2])
    y = np.sum(charge_coords[1] * charge_coords[2])
    x2 = np.sum(charge_coords[0] ** 2 * charge_coords[2])
    y2 = np.sum(charge_coords[1] ** 2 * charge_coords[2])
    xy = np.sum(charge_coords[0] * charge_coords[1] * charge_coords[2])
    x /= CHARGE
    y /= CHARGE
    x2 /= CHARGE
    y2 /= CHARGE
    xy /= CHARGE
    S2_x = x2 - x ** 2
    S2_y = y2 - y ** 2
    S_xy = xy - x * y
    d = S2_y - S2_x
    a = (d + np.sqrt(d ** 2 + 4 * S_xy ** 2)) / (2 * S_xy)
    b = y - a * x
    width = np.sqrt((S2_y + a ** 2 * S2_x - 2 * a * S_xy) / (1 + a ** 2))
    length = np.sqrt((S2_x + a ** 2 * S2_y + 2 * a * S_xy) / (1 + a ** 2))
    miss = np.abs(b / np.sqrt(1 + a ** 2))
    dis = np.sqrt(x ** 2 + y ** 2)
    q_coord = (x - charge_coords[0]) * (x / dis) + \
        (y - charge_coords[1]) * (y / dis)
    q = np.sum(q_coord * charge_coords[2]) / CHARGE
    q2 = np.sum(q_coord ** 2 * charge_coords[2]) / CHARGE
    azwidth = q2 - q ** 2
    return [width, length, miss, dis, azwidth]

# IRRADIATION IMAGES

# Read CNN output csv


# mean stdev median 1st quartile 3rd quartile min max
stats = {}

cnn_data = {}

headings = ["event_id", "p_noise", "p_spot",
            "p_track", "p_worm", "x_coord", "y_coord"]

with open("cnn_data.csv", "r") as cnn:
    reader = csv.reader(cnn)
    for row in reader:
        if(len(row) > 0):
            if(not "p_spot" in row):
                cnn_data[row[0]] = row[1:]

source_dir = "./irradiation_images/"

export_dir = "./results/irradiation/"

cnn_paths = []

for dir in os.listdir(source_dir):
    for imagefile in os.listdir(source_dir + dir):
        if(not imagefile[:-4] in cnn_data):
            cnn_paths.append(source_dir + dir + "/" + imagefile)

if(len(cnn_paths) > 0):
    from zoom_predictions_4class import run_blob_classifier

    cnn_weights_file = 'final_trained_weights.h5'
    df = run_blob_classifier(cnn_paths, 'out.csv', 4,
                             weights_file=cnn_weights_file)

    for i in range(len(df.index)):
        cnn_data[str(df['event_id'][i])] = [df['image_file'][i], df['p_noise'][i], df['p_spot']
                                            [i], df['p_track'][i], df['p_worm'][i], df['x_coord'][i], df['y_coord'][i]]

# update cnn_data.csv
with open("cnn_data.csv", "w") as cnn:
    writer = csv.writer(cnn)
    writer.writerow(headings)
    for id in cnn_data:
        writer.writerow([id] + cnn_data[id])

all_widths = []
all_lengths = []
all_azwidths = []

for dir in os.listdir(source_dir):

    hillas_params = []

    widths = []
    lengths = []
    azwidths = []

    for imagefile in os.listdir(source_dir + dir):
        image = imread(source_dir + dir + "/" + imagefile, flatten=True)

        charge_x = []
        charge_y = []
        charges = []

        # We run the CNN on each image. We save the CNN output to a csv, and check the file so that
        # duplicate CNN runs on the same image don't happen

        # We get the 'center'/origin coordinate from the CNN output

        origin = [0, 0]

        event_id = imagefile[:-4]

        origin[0] = int(cnn_data[event_id][5])
        origin[1] = int(cnn_data[event_id][6])

        radius_threshold = 40  # pixels, ~1um/pixel
        luminosity_threshold = 50  # unitless, ~deposited charge

        brights = np.argwhere(image > luminosity_threshold)

        for n in range(brights.shape[0]):
            x = brights[n, 0]
            y = brights[n, 1]

            if(y - origin[0] < radius_threshold and x - origin[1] < radius_threshold):
                charge_x.append(x-origin[1])
                charge_y.append(y-origin[0])
                charges.append(image[x, y])

        '''for n in range(height):
			for m in range(length):
				#pixel = image[n,m]
				
				#print(image)
				#print(image[n])
				
				if(image[n,m] > threshold):
					pixel = image[n,m]
					charge_x.append(m)
					charge_y.append(n)
					charges.append(pixel)'''

        event_hillas_params = hillas(
            [np.asarray(charge_x), np.asarray(charge_y), np.asarray(charges)])

        invalid = False
        for num in event_hillas_params:
           if np.isnan(num):
                invalid = True

        if(not invalid):
            hillas_params.append([event_id] + event_hillas_params)
            widths.append(event_hillas_params[0])
            lengths.append(event_hillas_params[1])
            azwidths.append(event_hillas_params[4])

    with open(export_dir + dir + ".csv", 'w') as out:
        writer = csv.writer(out)
        writer.writerow(["event id", "width", "length",
                         "miss", "dis", "azwidth"])
        for p in hillas_params:
            writer.writerow(p)

    all_widths += widths
    all_lengths += lengths
    all_azwidths += azwidths

    variables = {"width": widths, "length": lengths, "azwidth": azwidths}

    stats[dir] = {}

    # make stats summary for each variable
    for name in variables:
        var = variables[name]
        stats[dir][name] = [np.mean(var), np.std(
            var)] + [np.min(var)] + list(np.percentile(var, [25, 50, 75])) + [np.max(var)]

    # Make plots of parameter distributions by directory (classification)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
    plt.xscale('log')
    plt.hist(widths, bins=bins, density=True)
    plt.ylabel("Probability Density (1/pixels)")
    plt.xlabel("WIDTH (pixels, ~um)")
    plt.title("Distribution of WIDTH for '"+dir+"' Events (Irradiation)")
    # plt.hist(widths,25)

    plt.savefig('./results/irradiation/plots/'+dir+"/width_dist.png")

    plt.close()

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
    plt.xscale('log')
    plt.hist(lengths, bins=bins, density=True)
    plt.ylabel("Probability Density (1/pixels)")
    plt.xlabel("LENGTH (pixels, ~um)")
    plt.title("Distribution of LENGTH for '"+dir+"' Events (Irradiation)")
    # plt.hist(widths,25)

    plt.savefig('./results/irradiation/plots/'+dir+"/length_dist.png")

    plt.close()

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
    plt.xscale('log')
    plt.hist(azwidths, bins=bins, density=True)
    plt.ylabel("Probability Density (1/pixels)")
    plt.xlabel("AZWIDTH (pixels, ~um)")
    plt.title("Distribution of AZWIDTH for '"+dir+"' Events (Irradiation)")
    # plt.hist(widths,25)

    plt.savefig('./results/irradiation/plots/'+dir+"/azwidth_dist.png")

    plt.close()

# cumulative stats
variables_all = {"width": all_widths,
                 "length": all_lengths, "azwidth": all_azwidths}
stats["All"] = {}
for name in variables_all:
    var = variables_all[name]
    stats["All"][name] = [np.mean(var), np.std(
        var)] + [np.min(var)] + list(np.percentile(var, [25, 50, 75])) + [np.max(var)]

# Output to stats.csv

with open("./results/irradiation/stats.csv", 'w') as statsfile:
    writer = csv.writer(statsfile)
    for category in stats:
        writer.writerow([category])
        writer.writerow(["", "mean", "stdev", "min",
                         "Q1", "median", "Q3", "max"])
        for variable in stats[category]:
            writer.writerow([variable] + stats[category][variable])
        writer.writerow([])

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
plt.xscale('log')
plt.hist(all_widths, bins=bins, density=True)
plt.ylabel("Probability Density (1/pixels)")
plt.xlabel("WIDTH (pixels, ~um)")
plt.title("Distribution of WIDTH for All Events (Irradiation)")
plt.savefig("./results/irradiation/plots/All/width_dist.png")

plt.close()

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
plt.xscale('log')
plt.hist(all_lengths, bins=bins, density=True)
plt.ylabel("Probability Density (1/pixels)")
plt.xlabel("LENGTH (pixels, ~um)")
plt.title("Distribution of LENGTH for All Events (Irradiation)")
plt.savefig("./results/irradiation/plots/All/length_dist.png")

plt.close()

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
plt.xscale('log')
plt.hist(all_azwidths, bins=bins, density=True)
plt.ylabel("Probability Density (1/pixels)")
plt.xlabel("AZWIDTH (pixels, ~um)")
plt.title("Distribution of AZWIDTH for All Events (Irradiation)")
plt.savefig("./results/irradiation/plots/All/azwidth_dist.png")

plt.close()


# outdir ./results

#charge_coords = [[x_coords], [y_coords], [charges]]
