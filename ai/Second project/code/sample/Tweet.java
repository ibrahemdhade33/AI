package sample;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.MissingFormatArgumentException;
import java.util.ResourceBundle;

public class Tweet implements Initializable {



        String feature_selected ="" ;
        boolean fs =false ;

        @FXML
        private ComboBox Features;

        @FXML
        private TextField actions;

        @FXML
        private TextField followers;

        @FXML
        private TextField following;

        @FXML
        private TextField isretweet;

        @FXML
        private TextField location;

        @FXML
        private TextField result;

        @FXML
        private TextField tweettext;

        @FXML
        void FeaturesActions(ActionEvent event) {
            if (Features.getSelectionModel().getSelectedItem().toString().equals("All Features"))
                feature_selected = "af" ;
            else if (Features.getSelectionModel().getSelectedItem().toString().equals("Text Features"))
               feature_selected="tf" ;
            else
                feature_selected="nbf" ;
            fs = true ;
        }

        @FXML
        void resultaction(ActionEvent event) {
            try
            {
                if (!fs)
                    throw new MissingFormatArgumentException("you should choose the feature") ;
                else if (Features.getSelectionModel().getSelectedItem().toString().equals("All Features"))
                {
                    if (tweettext.getText().equals("") || followers.getText().equals("") ||
                    following.getText().equals("") || isretweet.getText().equals("")||
                    location.getText().equals("") || actions.getText().equals(""))
                        throw new MissingFormatArgumentException("you should fill all the fields") ;
                    else if (!isretweet.getText().equals("0") && !isretweet.getText().equals("1"))
                        throw new MissingFormatArgumentException("is retweet should be 0 or 1 only") ;
                    else if (!check_is_digits())
                        throw new MissingFormatArgumentException("follower and following and actions should be only digits") ;
                    else {
                        ProcessBuilder builder =new ProcessBuilder("C:\\Users\\Ibrah\\Desktop\\Machinelearning Script\\dist\\main.exe",
                                "tweet",feature_selected,tweettext.getText(),following.getText(),
                                followers.getText(),actions.getText(),isretweet.getText(),location.getText());

                        Process p = builder.start() ;
                        p.waitFor() ;
                        BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
                        BufferedReader b1  = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                        String line = null;
                        String line1 = null;

                        String s ="" ;
                        while ((line = b.readLine()) !=null){
                            s+=line ;
                        }
                        if (s.equals("0"))
                            result.setText("Ham");
                        else
                            result.setText("Spam");
                        while ((line1 = b1.readLine()) !=null){
                            System.out.println(line1);
                        }
                    }
                }
                else if (Features.getSelectionModel().getSelectedItem().toString().equals("Text Features")){
                    if (tweettext.getText().equals(""))
                        throw new MissingFormatArgumentException("you should enter the tweet text") ;
                    ProcessBuilder builder =new ProcessBuilder("C:\\Users\\Ibrah\\Desktop\\Machinelearning Script\\dist\\main.exe"
                            ,"tweet",feature_selected,tweettext.getText());

                    Process p = builder.start() ;
                    p.waitFor() ;
                    BufferedReader b  = new BufferedReader(new InputStreamReader(p.getInputStream()));
                    BufferedReader b1  = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                    String line = null;
                    String line1 = null;
                    String s ="" ;
                    while ((line = b.readLine()) !=null){
                        s+=line ;
                    }
                    if (s.equals("0"))
                        result.setText("Ham");
                    else
                        result.setText("Spam");
                    while ((line1 = b1.readLine()) !=null){
                        System.out.println(line1);
                    }
                }
                else if (Features.getSelectionModel().getSelectedItem().toString().equals("Naive Bias Features")){

                    if (tweettext.getText().equals(""))
                        throw new MissingFormatArgumentException("you should enter the tweet text") ;
                    ProcessBuilder builder =new ProcessBuilder("C:\\Users\\Ibrah\\Desktop\\Machinelearning Script\\dist\\main.exe"
                            ,"tweet",feature_selected,tweettext.getText());

                    Process p = builder.start() ;
                    p.waitFor() ;
                    BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
                    BufferedReader b1  = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                    String line =null;
                    String line1 = null;
                    String s ="";
                    while ((line = b.readLine()) !=null){
                            s+=line ;
                    }
                    if (s.equals("0"))
                        result.setText("Ham");
                    else
                        result.setText("Spam");
                    while ((line1 = b1.readLine()) !=null){
                        System.out.println(line1);
                    }

                }
            }
            catch (MissingFormatArgumentException | IOException | InterruptedException e){
                Alert alertCreat = new Alert(Alert.AlertType.ERROR);
                alertCreat.setTitle("Error");
                alertCreat.setHeaderText(null);
                alertCreat.setContentText(e.getMessage());
                alertCreat.showAndWait();
            }
        }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        ObservableList<String> Typefeature =  FXCollections.observableArrayList("All Features" ,"Text Features" ,"Naive Bias Features") ;
        Features.setItems(Typefeature);
    }
    boolean check_is_digits(){
        if (!following.getText().matches("[0-9]+") && !followers.getText().matches("[0-9]+")
        && actions.getText().matches("[0-9]+"))
            return false ;
        return true ;
    }
}


