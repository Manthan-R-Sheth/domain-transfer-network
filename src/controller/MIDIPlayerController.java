package controller;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;

import player.MIDIPlayer;
import view.MIDIPlayerView;

/**
 * Created with IntelliJ IDEA.
 * User: Administrator
 * Date: 11/22/13
 * Time: 11:57 AM
 * To change this template use File | Settings | File Templates.
 */
public class MIDIPlayerController {
    private MIDIPlayerView _view;
    private MIDIPlayer _model;

    /**
     * Instantiates the controller with a model and view.
     * @param view
     * @param model
     */
    public MIDIPlayerController(MIDIPlayerView view, MIDIPlayer model) {
        _view = view;
        _model = model;

        setUpActionListeners();
    }

    /**
     * Sets up the controls for the application.
     */
    private void setUpActionListeners() {
        setUpPlayerListeners();
    }

    /**
     * Sets up the controls for the MIDI Player.
     */
    private void setUpPlayerListeners() {
        addFileOpenListener();
        addFileSaveListener();
        addPNGMusicListener();
        addPlayListener();
        addPauseListener();
        addStopListener();
        addBackListener();
        addForwardListener();
    }

    /**
     * Control for the open button.
     */
    private void addFileOpenListener() {
        _view.addFileOpenListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                	_view.openFileChooser();
                	String filename = _view.getSelectedFile().getName();
                	_model.open(_view.getSelectedFile());
                    _view.setStatusFieldText("Opening " + filename + "...");
                    //_model.open(filename);
                    _model.stop();
                    _view.setStatusFieldText("Opened " + filename);
                }
                catch(Exception exception) {
                	_view.setStatusFieldText("Failed to open file");
                    _view.displayMessageBox(exception.toString());
                }
            }
        });
    }

    /**
     * Control for the save button.
     */
    private void addFileSaveListener() {
        _view.addPlayerFileSaveListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                	File existingFile = _view.getSelectedFile();
                	_view.openFileSaver();
                	String filename = _view.getSelectedFile().getName();
                	if(existingFile == null ||
                			!existingFile.exists())
                		throw new Exception("No file loaded!");
                	else if(!existingFile.getName().substring(existingFile.getName().length()-4).equals(filename.substring(filename.length()-4)))
                		throw new Exception("Invalid file type!");
                    _view.setStatusFieldText("Saving " + filename);
                    _model.save(filename);
                    _view.setStatusFieldText("Saved " + filename);
                }
                catch(Exception exception) {
                	_view.setStatusFieldText("Failed to save file");
                    _view.displayMessageBox(exception.toString());
                }
            }
        });
    }

    /**
     * Control for the PNGMusic button.
     */
    private void addPNGMusicListener() {
        _view.addPlayerPNGMusicListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    BufferedImage image = _model.convertSequenceToPNG();
                    _view.setStatusFieldText("Converting to PNG...");
                    _view.displayImage(image);
                    _view.setStatusFieldText("Converted to PNG");
                }
                catch(Exception exception) {
                	//Doesn't do anything.
                }
            }
        });
    }

    /**
     * Control for the play button.
     */
    private void addPlayListener() {
        _view.addPlayerPlayListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    _model.resume();
                    _view.setStatusFieldText("Playing");
                }
                catch(Exception exception) {
                    try {
                        _model.play();
                    }
                    catch(Exception exception2) {
                        _view.displayMessageBox(exception2.toString());
                    }
                }
            }
        });
    }

    /**
     * Control for the pause button.
     */
    private void addPauseListener() {
        _view.addPlayerPauseListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    _model.pause();
                    _view.setStatusFieldText("Paused");
                } catch (Exception exception) {
                    //Do nothing.
                }
            }
        });
    }

    /**
     * Control for the stop button.
     */
    private void addStopListener() {
        _view.addPlayerStopListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    _model.stop();
                    _view.setStatusFieldText("Stopped");
                } catch (Exception exception) {
                    //Do nothing.
                }
            }
        });
    }

    /**
     * Control for the back button.
     */
    private void addBackListener() {
        _view.addPlayerBackListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    _model.stop();
                    _model.play();
                    _view.setStatusFieldText("Playing");
                } catch (Exception exception) {
                    //Do nothing.
                }
            }
        });
    }

    /**
     * Control for the forward button.
     */
    private void addForwardListener() {
        _view.addPlayerForwardListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    _model.stop();
                    _view.setStatusFieldText("Stopped");
                } catch (Exception exception) {
                    //Do nothing.
                }
            }
        });
    }
  
//    public static void main(String[] args) throws Exception {
//    	MIDIPlayerView v = new MIDIPlayerView();
//    	MIDIPlayer m = new MIDIPlayer();
//    	MIDIPlayerController abc = new MIDIPlayerController(v, m);
//    }
    
}
