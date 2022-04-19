package sample;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.MissingFormatArgumentException;
import java.util.ResourceBundle;


public class Controller implements Initializable {

    HashMap<String,Boolean>hashMap = new HashMap<>() ;//hasp map tp prevent the user from plotting a chart that already plotted
    String model = "" ;//for the chosen model
    String feature ="" ; //for the chosen feature
    boolean model_selected =false ;//to check if the user select model or not
    boolean feature_selected=false ;//to check if the user select a feature or not
    ArrayList <String>results = new ArrayList<>() ;//to store the result string
    ObservableList<table> list = FXCollections.observableArrayList();//for the table

    @FXML
    private ComboBox chosefeature;

    @FXML
    private TextField resulttext;
    @FXML
    private TextField acurecy;
    @FXML
    private TextField tweettext;
    @FXML
    private TableColumn<table, String> actualham;

    @FXML
    private TableColumn<table, String> actualspam;
    @FXML
    private TableColumn<table, String> ccc;
    @FXML
    private TableView<table> table;
    @FXML
    private TextField precsion;

    @FXML
    private TextField recall;


    @FXML
    void chsefeatureaction(ActionEvent event) {
        //actions in combobox for features
        if (chosefeature.getSelectionModel().getSelectedItem().toString().equals("All Features")){
            feature ="af" ;
        }
        else if (chosefeature.getSelectionModel().getSelectedItem().toString().equals("Best Features")){
            feature="ba" ;
        }
        else if(chosefeature.getSelectionModel().getSelectedItem().toString().equals("Text Features")){
            feature="at" ;
        }
        else{
            feature ="bt" ;
        }
        feature_selected =true ;
    }

        @FXML
        private TextField browsesamplesdatatext;

        @FXML
        private TextField browsetweetdatattext;

        @FXML
        void brosesampledata(ActionEvent event) {
            //browse sample data button
            FileChooser fileChooserShares = new FileChooser();
            fileChooserShares.setTitle("Select project file ");
            fileChooserShares.getExtensionFilters().addAll(
                    new FileChooser.ExtensionFilter("Text Files", "*.csv"));

            File selectedFile = fileChooserShares.showOpenDialog(null);
            if (String.valueOf(selectedFile).equals("null")) {
                return;
            } else {
                browsesamplesdatatext.setText(selectedFile.toString());
            }
        }

        @FXML
        void browsetweetdata(ActionEvent event) {
            //browse tweet data button
            FileChooser fileChooserShares = new FileChooser();
            fileChooserShares.setTitle("Select project file ");
            fileChooserShares.getExtensionFilters().addAll(
                    new FileChooser.ExtensionFilter("Text Files", "*.csv"));

            File selectedFile = fileChooserShares.showOpenDialog(null);
            if (String.valueOf(selectedFile).equals("null")) {
                return;
            } else {
                browsetweetdatattext.setText(selectedFile.toString());
            }
        }



    @FXML
    void viewtweettestpage(ActionEvent event) throws IOException {
            //to view the page to test a single tweet
        Parent root = FXMLLoader.load(getClass().getResource("tweet.fxml"));
        Stage stage3 = new Stage();
        stage3.setScene(new Scene(root));
        stage3.setTitle("Single Tweet Test");
        stage3.showAndWait();
    }


    @FXML
    private ComboBox chosemodel;

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        //this function run when the program start and set the items in both combo box and set all the feature as not plotted
        //set the items in both combobox
        ObservableList<String> models =  FXCollections.observableArrayList("Decision Tree" ,"Naive bias","Neural Network") ;
        chosemodel.setItems(models);
        ObservableList<String> Typefeature =  FXCollections.observableArrayList("All Features" ,"Best Features","Text Features" ,"Best Text Features") ;
        //set all the features as not plotted
        chosefeature.setItems(Typefeature);
        hashMap.put("All Features",false) ;
        hashMap.put("Best Features",false);
        hashMap.put("Text Features",false);
        hashMap.put("Best Text Features",false) ;

    }

    @FXML
    void chosemodelaction(ActionEvent event) {
        //for chosen item from the model combobox
        if (chosemodel.getSelectionModel().getSelectedItem().toString().equals("Decision Tree")){
            model = "tree" ;
        }
        else if (chosemodel.getSelectionModel().getSelectedItem().toString().equals("Naive bias")){
            model ="naive_bias" ;
        }
        else {
            model = "network" ;
        }
        model_selected = true ;
    }


    @FXML
    void TestingdatatAction(ActionEvent event) throws IOException, InterruptedException {
        try{
            //check if the user chose the model and feature
            if (!checkvalid() )
                throw new MissingFormatArgumentException("please choose the model and features");
            list.clear();
            results.clear();
           //creat the process builder and set the arguments and the path for executable file on it
            ProcessBuilder builder = new ProcessBuilder("C:\\Users\\Ibrah\\Desktop\\Machinelearning Script\\dist\\main.exe","atf",model,feature) ;
           //run the process builder
            Process p = builder.start() ;
            p.waitFor() ;
            //read the data from python script console it by buffer reader using process input stream
            BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
            ////read the errors from python script console it by buffer reader using process error stream
            BufferedReader b1 = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            String line = null;
            String line1 = null;
            //read the data line by line
            while ((line = b.readLine()) !=null){
                results.add(line) ;
            }
            //read the errors line by line
            while ((line1 = b1.readLine()) !=null){
                System.out.println(line1);
            }

            String[]res = new String[3] ;

            for (int i = 0 ; i < 3 ; i++){
                String s= results.get(i) ;
                String []s1 = s.split(" ") ;
                res[i] = s1[1] ;
            }
            //split the string and get the results
            acurecy.setText(res[0]);
            recall.setText(res[1]);
            precsion.setText(res[2]);
            String []conf = new String[4] ;
            for (int i = 4,j=0 ; i < 8 ;i++,j++){
                String s= results.get(i) ;
                String []s1 = s.split(" ") ;
                conf[j] = s1[2] ;
            }
            //print the confusion matrix in the table
            list.add(new table("Predicted Spam" ,conf[0],conf[1])) ;
            list.add(new table("Pridected Ham" ,conf[2],conf[3])) ;
            ccc.setCellValueFactory(new PropertyValueFactory<>("c1"));
            actualspam.setCellValueFactory(new PropertyValueFactory<>("c2"));
            actualham.setCellValueFactory(new PropertyValueFactory<>("c3"));
            table.setItems(list);
        }
        //show the thrown exception using alert
        catch(MissingFormatArgumentException e){
            Alert alertCreat = new Alert(Alert.AlertType.ERROR);
            alertCreat.setTitle("Error");
            alertCreat.setHeaderText(null);
            alertCreat.setContentText(e.getMessage());
            alertCreat.showAndWait();
        }

    }
    @FXML
    void samplesdaataresults(ActionEvent event) throws IOException, InterruptedException {
        try{
            //check that the user choose the feature and model and browse the file
            if (!checkvalid() || browsesamplesdatatext.getText().equals(""))
                throw new MissingFormatArgumentException("please browse file for sample data or choose model and feature");
            results.clear();
            list.clear();

            ProcessBuilder builder = new ProcessBuilder("C:\\Users\\Ibrah\\Desktop\\Machinelearning Script\\dist\\main.exe","adf",model,feature,browsesamplesdatatext.getText()) ;
            Process p = builder.start() ;
            p.waitFor() ;
            BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader b1 = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            String line = null;
            String line1 = null;

            while ((line = b.readLine()) !=null){
                results.add(line) ;
            }
            while ((line1 = b1.readLine()) !=null){
                System.out.println(line1);
            }


            String[]res = new String[3] ;

            for (int i = 0 ; i < 3 ; i++){
                String s= results.get(i) ;
                String []s1 = s.split(" ") ;
                res[i] = s1[1] ;
            }
            acurecy.setText(res[0]);
            recall.setText(res[1]);
            precsion.setText(res[2]);
            String []conf = new String[4] ;
            for (int i = 4,j=0 ; i < 8 ;i ++,j++){
                String s= results.get(i) ;
                String []s1 = s.split(" ") ;
                conf[j] = s1[2] ;
            }
            list.add(new table("Predicted Spam" ,conf[0],conf[1])) ;
            list.add(new table("Pridected Ham" ,conf[2],conf[3])) ;
            ccc.setCellValueFactory(new PropertyValueFactory<>("c1"));
            actualspam.setCellValueFactory(new PropertyValueFactory<>("c2"));
            actualham.setCellValueFactory(new PropertyValueFactory<>("c3"));
            table.setItems(list);
        }
        catch (MissingFormatArgumentException e){
            Alert alertCreat = new Alert(Alert.AlertType.ERROR);
            alertCreat.setTitle("Error");
            alertCreat.setHeaderText(null);
            alertCreat.setContentText(e.getMessage());
            alertCreat.showAndWait();
        }

    }
    boolean checkvalid(){
        if (model_selected && feature_selected)
            return true ;
        return false ;
    }
    @FXML
    void tweetsfileresults(ActionEvent event) {
        try{
            if (browsetweetdatattext.getText().equals(""))
                throw new MissingFormatArgumentException("please browse file for sample data or choose model and feature");
            results.clear();
            list.clear();
            ProcessBuilder builder = new ProcessBuilder("C:\\Users\\Ibrah\\Desktop\\Machinelearning Script\\dist\\main.exe","anf",browsetweetdatattext.getText()) ;
            Process p = builder.start() ;
            p.waitFor() ;
            BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader b1 = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            String line = null;
            String line1 = null;

            while ((line = b.readLine()) !=null){
               results.add(line) ;
            }
            while ((line1 = b1.readLine()) !=null){
                System.out.println(line1);
            }


            String[]res = new String[3] ;

            for (int i = 0 ; i < 3 ; i++){
                String s= results.get(i) ;
                String []s1 = s.split(" ") ;
                res[i] = s1[1] ;
            }
            acurecy.setText(res[0]);
            recall.setText(res[1]);
            precsion.setText(res[2]);
            String []conf = new String[4] ;
            for (int i = 4,j=0 ; i < 8 ;i ++,j++){
                String s= results.get(i) ;
                String []s1 = s.split(" ") ;
                conf[j] = s1[2] ;
            }
            list.add(new table("Predicted Spam" ,conf[0],conf[1])) ;
            list.add(new table("Pridected Ham" ,conf[2],conf[3])) ;
            ccc.setCellValueFactory(new PropertyValueFactory<>("c1"));
            actualspam.setCellValueFactory(new PropertyValueFactory<>("c2"));
            actualham.setCellValueFactory(new PropertyValueFactory<>("c3"));
            table.setItems(list);
        }
        catch (MissingFormatArgumentException | IOException | InterruptedException e){
            Alert alertCreat = new Alert(Alert.AlertType.ERROR);
            alertCreat.setTitle("Error");
            alertCreat.setHeaderText(null);
            alertCreat.setContentText(e.getMessage());
            alertCreat.showAndWait();
        }
    }
    @FXML
    private BarChart<String, Double> barchart;
    @FXML
    void plotig(ActionEvent event) {
        try{
            if (!feature_selected)
                throw new MissingFormatArgumentException("please choose the features");
            else if (hashMap.get(chosefeature.getSelectionModel().getSelectedItem().toString()))
                throw new MissingFormatArgumentException("the chart is already plotted") ;
            results.clear();

            ProcessBuilder builder = new ProcessBuilder("C:\\Users\\Ibrah\\Desktop\\Machinelearning Script\\dist\\main.exe"
                    ,"ig",feature) ;
            Process p = builder.start() ;
            p.waitFor() ;
            BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader b1 = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            String line = null;
            String line1 = null;

            while ((line = b.readLine()) !=null){
                results.add(line) ;
            }
            while ((line1 = b1.readLine()) !=null){
                System.out.println(line1);
            }



            XYChart.Series<String,Double> series = new XYChart.Series<>() ;

            series.setName(chosefeature.getSelectionModel().getSelectedItem().toString());
            hashMap.put(chosefeature.getSelectionModel().getSelectedItem().toString(),true);
            for (int i =0 ; i < results.size();i++){
                String[]s = results.get(i).split(" ");
                double ig = Double.parseDouble(s[1]) ;
                series.getData().add(new XYChart.Data<>(s[0],ig));
            }
            barchart.getData().add(series) ;
        }
        catch (MissingFormatArgumentException | IOException | InterruptedException e){
            Alert alertCreat = new Alert(Alert.AlertType.ERROR);
            alertCreat.setTitle("Error");
            alertCreat.setHeaderText(null);
            alertCreat.setContentText(e.getMessage());
            alertCreat.showAndWait();
        }
    }

}


