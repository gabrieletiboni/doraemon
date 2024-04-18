import jinja2
import os
import numpy as np
from math import sin, cos

class TemplateRenderer():
    """
    The class for rendering Jinja templates. The XML files are searched for in
    the `franka_sim/templates` directory, as well as any  directory included in
    the ``FRANKA_TEMPLATE_PATH`` environment variable (multiple directories can
    be provided, separated by a colon ``:``). Meshes are searched for in
    `franka_sim/meshes`, as well as directories from ``FRANKA_MESH_PATH``
    (note that multiple mesh directories have not been tested properly and this
    feature may not work as intended)
    """
    def __init__(self):
        # Set the relative template dir as base
        base_dir = os.path.dirname(__file__)
        template_dir = os.path.join(base_dir, "..", "franka_sim")
        self.template_dirs = [template_dir]

        # Get additional template dirs from an env variable
        if "FRANKA_TEMPLATE_PATH" in os.environ:
            dirs = os.environ["FRANKA_TEMPLATE_PATH"].split(":")
            self.template_dirs.extend(dirs)

        # And do the same for meshes
        mesh_dir = os.path.join(base_dir, "..", "franka_sim", "assets")
        self.mesh_dirs = [mesh_dir]

        if "FRANKA_MESH_PATH" in os.environ:
            dirs = os.environ["FRANKA_MESH_PATH"].split(":")
            self.mesh_dirs.extend(dirs)

        self.loader = jinja2.FileSystemLoader(searchpath=self.template_dirs)
        self.template_env = jinja2.Environment(loader=self.loader)

    def render_template(self, template_file, **kwargs):
        """
        :description: This function renders an XML template and returns the
            resulting XML.
        :param template_file: name of the XML file, relative to the
            `franka_sim/templates` directory or any directory from the
            ``FRANKA_MESH_PATH`` env variable
        :param \*\*kwargs: any keyword arguments will be passed to the template.
        :return: the rendered XML data
        """
        template = self.template_env.get_template(template_file)
        rendered_xml = template.render(mesh_dirs=self.mesh_dirs, pi=np.pi,
                                       sin=sin, cos=cos, **kwargs)
        return rendered_xml

    def render_to_file(self, template_file, target_file, **kwargs):
        """
        :description: Renders a template to a file.
        :param template_file: XML template file name to be passed to
            ``render_template``,
        :param target_file: the path where the rendered XML file will be saved,
        :param \*\*kwargs: any keyword arguments will be passed to the template.
        """
        xml = self.render_template(template_file, **kwargs)
        with open(target_file, "w") as f:
            f.write(xml)

