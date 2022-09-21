from datetime import datetime
from os.path import join, exists
from os import makedirs

from utils import data_path


def log_experiment(description="",
                   tag="",
                   version="1.0",
                   number=1,
                   folder=data_path('logs'),
                   records={}):
    """
    This package contains simple methods to create a starlog (from StarTrek).
    The Starlog is a recording entered into a starship computer record
    for posterity. The log was used to inform the captain's superiors
    of what was happening on a
    mission and to record historical facts for future generations.

    In other words, this library is going to help you to
    keep track the experiments you run, the context and its results.

    :param description the experiment definition, changes and
    expected/actual result
    :param tag a unique tag to classify the line of experimentation,
    like "ml" or "feateng"
    :param version the master version of the project
    :param number of the experiment, incremental. Fail if already was recorded
    :param folder (optional) to store all experiment records
    :param *args and **kwargs all elements you want to record

    :raises ValueError if the experiment was already recorded
    """
    # Star date
    stardate = datetime.now()

    # The destination folder must exist
    vfolder = join(folder, version)
    if not exists(vfolder):
        makedirs(vfolder)

    # Define the experiment file
    logfile = join(vfolder, "exp{}.{}.{}.txt".format(".{}".format(tag) if tag else "", version,
                                                     number))

    # If the file exist fail with an error.
    # We don't want to override a previous experiment information
    if exists(logfile):
        raise ValueError("File {} already exists."
                         "Manually delete the file and try again".format(logfile))

    # Define the summary
    summary = ""

    # Title, Description and Records formatting
    summary = "\nExperiment #{} (v:{}{})".format(number, version, "-{}".format(tag) if tag else "")
    summary = "{}\nStardate: {}".format(summary, stardate)
    summary = "{}\n\nCaptain's log:\n{}\n".format(summary, description)
    for key, value in records.items():
        summary = "{}\n{}:\n{}\n".format(summary, key, _entity_to_human(value))

    # Write the output
    with open(logfile, "w") as fp:
        fp.write(summary)

    # If everything was successfull, write an entry in the capitans log:
    write_capitan_log(stardate=stardate,
                      description=description,
                      version=version,
                      number=number,
                      folder=folder)


def write_capitan_log(stardate, description, version, number, folder):
    """
    Append the entry in the main capitan's log.
    """
    # Capitan's log filename
    logfile = join(folder, "capitan.log.txt")
    # Append the summary
    summary = "\n\n• Experiment v:{}.{} - Stardate:{}\n\t{}".format(version, number, stardate,
                                                                    description)
    # Push the changes to the master file
    with open(logfile, "a") as fp:
        fp.write(summary)


def _entity_to_human(x):
    """
    Map the entity into a human readable string.
    Accepted types:
    - Built-in types (str, int, double, float)
    - List
    - Pandas Dataframe
    """
    if 'pandas.core.frame.DataFrame' in str(type(x)):
        return x
    elif 'pandas.core.series.Series' in str(type(x)):
        return x
    elif 'pandas.core.indexes.base.Index' in str(type(x)):
        return [c for c in x]
    else:
        return x
