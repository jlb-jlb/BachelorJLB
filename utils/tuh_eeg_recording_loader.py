import os
import random

# import numpy as np


class DataLoader:
    def __init__(self, root_path, extensions=[], shuffle=True, seed=12071998):
        """
        Initialize the DataLoader object.

        This method initializes the DataLoader object with the specified root path, file extensions, shuffle flag, and seed.
        If no specific extensions are provided, all file extensions are considered.
        The shuffle flag and seed are not implemented yet.

        Parameters:
        root_path (str): The root directory to search for files.
        extensions (list): The list of file extensions to consider. Default is an empty list, which means all extensions are considered.
        shuffle (bool): Whether to shuffle the file tuples. Default is False. Not implemented yet.
        seed (int): The seed for the random number generator. Default is 0. Not implemented yet.

        Returns:
        None
        """

        self._shuffle = shuffle
        self._seed = seed

        self.file_tuples = []
        self.current_index = 0
        # norm root_path to operating system
        self.root_path = os.path.abspath(os.path.normpath(root_path))
        self.extensions = (
            extensions  # if empty list, all file extensions are considered
        )
        self.get_previous_active = False

        # Find all files in root_path
        self._find_files()

        if self._shuffle:
            # create generator that is seeded and generates random numbers in the range [0, len(file_tuples)).
            # It will generate each element in the range only once.
            self.random_generator = random.Random(self._seed).sample(
                range(len(self.file_tuples)), len(self.file_tuples)
            )
            self.random_index = 0

        self._description = self.get_description()

    def _find_files(self) -> None:
        """
        Find all files in the root directory.

        This method traverses the directory specified by root_path during the initialization of the DataLoader object, and identifies all unique base filenames.
        If specific extensions are provided, it creates a tuple of associated files for each base filename and checks if all files with those extensions exist.
        If not, it raises an error. If no specific extensions are provided, it gathers all files with the same base filename.

        This method is called during the initialization of the DataLoader object.

        Returns:
        None
        """

        # Traverse directory and identify unique base filenames
        for root, _, files in os.walk(self.root_path):
            base_filenames = set(os.path.splitext(f)[0] for f in files)
            for base_filename in base_filenames:
                if self.extensions:
                    # If specific extensions are provided, create tuple of associated files
                    # check if all files with those extensions exist, if not raise error
                    file_tuple = tuple(
                        os.path.join(root, base_filename + ext)
                        for ext in self.extensions
                    )
                    if all(os.path.exists(f) for f in file_tuple):
                        self.file_tuples.append(file_tuple)
                    else:
                        raise FileNotFoundError(f"File not found: {file_tuple}")
                else:
                    # If no specific extensions are provided, gather all files with the same base filename
                    matching_files = [
                        os.path.join(root, f)
                        for f in files
                        if os.path.splitext(f)[0] == base_filename
                    ]
                    if matching_files:
                        self.file_tuples.append(tuple(matching_files))

        # self.file_tuples = np.array(self.file_tuples)

    def get_current_file_tuple(self) -> tuple:
        """
        Get the current file tuple.

        This method returns the current file tuple based on the current index.

        returns: current file tuple
        """

        if self.current_index > 0 and self.get_previous_active is False:
            return self.file_tuples[self.current_index - 1]

        return self.file_tuples[self.current_index]

    def get_file_tuples(self) -> list:
        """
        Get the list of file tuples.

        This method returns the list of file tuples that have been identified in the directory specified by root_path during the initialization of the DataLoader object.

        returns: list of file tuples
        """

        return self.file_tuples

    def get_extensions(self) -> list:
        """
        Get the list of file extensions.

        This method returns the list of file extensions that were specified during the initialization of the DataLoader object.
        If no specific extensions were provided, it returns an empty list.

        returns: list of file extensions
        """

        return self.extensions

    def get_next(self) -> tuple:
        """
        Get the next file tuple.

        It increments the current index by 1 and returns the next file tuple.
        If the current index is greater than or equal to the length of the file tuples, it returns None.

        returns: next file tuple or None if no more files
        """

        if self.current_index >= len(self.file_tuples):
            return None  # no more files
        # Get next file tuple
        next_tuple = self.file_tuples[self.current_index]
        self.current_index += 1
        return next_tuple

    def get_previous(self) -> tuple:
        """
        Get the previous file tuple.

        It decrements the current index by 1 and returns the previous file tuple.
        If the current index is less than or equal to 0, it returns None.

        returns: previous file tuple or None if no more files
        """

        if self.current_index <= 0:
            return None
        elif self.get_previous_active is False:
            self.get_previous_active = True
            self.current_index -= 1
        # Get previous file tuple
        previous_tuple = self.file_tuples[self.current_index - 1]
        self.current_index -= 1
        return previous_tuple

    def reset(self) -> None:
        self.current_index = 0

    def get_next_shuffled(self) -> tuple:
        """
        Get the next shuffled file tuple.

        This method returns the next shuffled file tuple from the list of file tuples.
        If the shuffle flag is set to False, it raises a ValueError.
        If the random index is greater than or equal to the length of the file tuples, it returns None.

        Returns:
        next shuffled file tuple or None if no more files
        """

        if self._shuffle is False:
            raise ValueError(
                "Shuffle is set to False. Cannot get next shuffled file tuple."
            )
        if self.random_index >= len(self.file_tuples):
            return None
        # Get next shuffled file tuple
        next_tuple = self.file_tuples[self.random_generator[self.random_index]]
        self.random_index += 1
        return next_tuple

    def describe(self) -> str:
        """
        Describe the loaded data.

        This method returns a string that describes the loaded data. It denotes the different file extensions in each tuple, the length of the tuples, and the length of the file_tuples list.

        Returns:
        str: A description of the loaded data.
        """

        description = f"Number of file tuples: {len(self.file_tuples)}\n"
        description += f"Total number of files: {self.get_total_files()}\n"
        description += f"Unique extensions: {self.get_unique_extensions()}\n"
        description += f"Unique tuple lengths: {self.get_unique_tuple_length_counts()}"
        return description

    def get_description(self) -> str:
        """
        Describe the loaded data.

        This method returns a string that describes the loaded data. It denotes the different file extensions in each tuple, the length of the tuples, and the length of the file_tuples list.

        Returns:
        str: A description of the loaded data.
        """
        return {
            "file_tuples": len(self.file_tuples),
            "total_files": self.get_total_files(),
            "unique_extensions": self.get_unique_extensions(),
            "unique_tuple_lengths": self.get_unique_tuple_length_counts(),
        }

    def get_unique_tuple_length_counts(self) -> dict:
        """
        Get the counts of unique tuple lengths.

        This method returns a dictionary that contains the counts of unique tuple lengths in the file tuples. The keys of the dictionary are the unique tuple lengths, and the values are the corresponding counts.

        Returns:
        dict: The counts of unique tuple lengths.
        """
        unique_counts = {}
        for num in self.get_file_tuple_lengths():
            if num in unique_counts:
                unique_counts[num] += 1
            else:
                unique_counts[num] = 1

        return unique_counts

    def get_file_tuple_lengths(self) -> list:
        """
        Get the lengths of the file tuples.

        This method returns a list of the lengths of the file tuples.

        Returns:
        list: The lengths of the file tuples.
        """
        return [len(file_tuple) for file_tuple in self.file_tuples]

    def get_unique_extensions(self) -> set:
        """
        Get the unique file extensions.

        This method returns a set of the unique file extensions in the file tuples.

        Returns:
        set: The unique file extensions.
        """
        extensions = set()
        for file_tuple in self.file_tuples:
            for file in file_tuple:
                extension = os.path.splitext(file)[1]
                extensions.add(extension)
        return extensions

    def get_total_files(self) -> int:
        """
        Get the total number of files.

        This method returns the total number of files in the file tuples.

        Returns:
        int: The total number of files.
        """
        total_files = 0
        for file_tuple in self.file_tuples:
            total_files += len(file_tuple)
        return total_files
