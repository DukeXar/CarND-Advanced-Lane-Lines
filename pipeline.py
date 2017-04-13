class Processor(object):
    """
    Sits in the pipeline and processes incoming frames.
    """
    def __init__(self):
        self._output = None

    def process(self, image):
        """
        This is called by the pipeline to process the image, or other values (e.g. lane fit)
        """
        self._output = self.apply(image)

    def apply(self, image):
        """
        Should be overridden in the child class to run the actual processing
        Returns processed image or values (e.g. lane fits)
        """
        raise NotImplementedError()

    def dump_input_frame(self, image):
        """
        Useful to dump additional information on the input image for visualization purposes.
        """
        raise NotImplementedError()

    def dump_output_frame(self, image):
        """
        Useful to dump additional information on the processed image for visualization purposes.
        """
        raise NotImplementedError()

    @property
    def output(self):
        """
        Last processed image.
        """
        return self._output