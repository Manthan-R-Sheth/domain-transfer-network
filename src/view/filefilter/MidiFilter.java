package view.filefilter;

import java.io.File;

import javax.swing.filechooser.FileFilter;

import tools.ImageAndMusicTools;

/**
 * A FileFilter that filters out only MIDI files.
 * @author abhishekchatterjee
 * @date Dec 17, 2015
 * @time 3:10:00 PM
 */
public class MidiFilter extends FileFilter {

	@Override
	public boolean accept(File f) {
		// TODO Auto-generated method stub
		if (f.isDirectory()) {
	        return true;
	    }

	    String extension = ImageAndMusicTools.getExtension(f);
	    if (extension != null) {
	        if (extension.equals("mid"))
	                return true;
	        else return false;
	    }
	    return false;
	}

	@Override
	public String getDescription() {
		// TODO Auto-generated method stub
		return null;
	}

}
