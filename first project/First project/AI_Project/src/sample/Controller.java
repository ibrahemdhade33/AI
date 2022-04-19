package sample;


import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;


public class Controller {
    public static int fi = 0, sec = 0, th = 0, other = 0;
    static ArrayList<chart> charts = new ArrayList<>();
    ObservableList<tabelView> list = FXCollections.observableArrayList();
    Boolean checkSubmit = false;
    @FXML
    private Button btnSubmit;
    @FXML
    private TextField txtNumLoop;
    @FXML
    private TextField txtNumRun;
    @FXML
    private TextField txtPathFile;
    @FXML
    private Button btnBrowse;
    @FXML
    private TableColumn<tabelView, Integer> alg;
    @FXML
    private TableColumn<tabelView, Integer> first;
    @FXML
    private TableColumn<tabelView, String> groups;
    @FXML
    private TableColumn<tabelView, Integer> second;
    @FXML
    private TableColumn<tabelView, Integer> third;
    @FXML
    private TableView<tabelView> tabel;
    @FXML
    private TableColumn<tabelView, Integer> number;

    @FXML
    void showchart(ActionEvent event) {

        try {
            if (checkSubmit) {
                Parent root = FXMLLoader.load(getClass().getResource("chart.fxml"));
                Stage stage2 = new Stage();
                stage2.setScene(new Scene(root));
                stage2.show();


            } else {
                throw new IllegalAccessException("Pleas fill the information and submit");
            }
        } catch (IllegalArgumentException | IOException | IllegalAccessException argumentException) {

            Alert alertCreat = new Alert(Alert.AlertType.WARNING);
            alertCreat.setTitle("Warning");
            alertCreat.setHeaderText(null);
            alertCreat.setContentText(argumentException.getMessage());
            alertCreat.showAndWait();

        }

    }


    @FXML
    void btnSubmitAction(ActionEvent event) throws Exception {


        try {

            if (txtNumLoop.getText().matches("[0-9]+")) {

            } else if (txtNumLoop.getText().equals("")) {

                throw new IllegalArgumentException("Pleas enter number of iteration");

            } else {

                throw new IllegalArgumentException("Pleas enter digits only !!");

            }

            if (!txtPathFile.getText().equals("")) {
                charts.clear();
                fi = 0;
                sec = 0;
                th = 0;
                other = 0;

                //array list to store the information from the file
                ArrayList<info> arr = new ArrayList<info>();
                //read the file as csv file
                BufferedReader readFile = new BufferedReader(new FileReader(txtPathFile.getText()));
                String readFilerow;
                readFilerow = readFile.readLine();
                while ((readFilerow = readFile.readLine()) != null) {
                    String[] data = readFilerow.split(",");
                    String temp = "";
                    if (data.length == 6) {
                        temp = data[0] + "," + data[1] + "," + data[2];
                        arr.add(new info(temp, Integer.parseInt(data[3]), Integer.parseInt(data[4]), Integer.parseInt(data[5])));
                    } else if (data.length == 5) {
                        temp = data[0] + "," + data[1];
                        arr.add(new info(temp, Integer.parseInt(data[2]), Integer.parseInt(data[3]), Integer.parseInt(data[4])));
                    }
                }
                //creat a Set for the projects chosen by all groups
                Set<Integer> s = new HashSet<>();
                for (int i = 0; i < arr.size(); i++) {
                    s.add(arr.get(i).getF());
                    s.add(arr.get(i).getS());
                    s.add(arr.get(i).getT());
                }
                //convert the set to array list
                ArrayList<Integer> t = new ArrayList<>(s);
                ArrayList<Integer> r = new ArrayList<>();
                //creat list for all projects in the projects file
                for (int i = 1; i < 39; i++)
                    r.add(i);
                //creat an object of initial solution class and send the two lists to it
                initialsol e = new initialsol(t, r, arr.size());
                //creat array list for initial solutions
                ArrayList<int[]> is = new ArrayList<>();
                System.out.println(t.size() + " " + r.size());
                //creat 10 initial solutions
                for (int i = 0; i < 10; i++)
                    is.add(e.sol());
                //creat an object from gentic algorithim class
                geniticalgorithim p = new geniticalgorithim(is, arr);
                //sort the initial solution based on solution fitness function
                is = p.bubblesort();
                //update the value for initial solution list in the genitic object
                p.setArr1(is);

                //define loop iteration variable which get the value from the text field in the interface
                int loopiter = Integer.parseInt(txtNumLoop.getText().trim());
                //start gentic algorithm loop
                for (int i = 0; i < loopiter; i++) {

                    //creat object for random number
                    Random random = new Random();

                    //Array list to store the result for Crossover operation
                    ArrayList<int[]> crossresut = p.cross_over();
                    //Add the result of cross over operation to the population
                    is.add(crossresut.get(0));
                    is.add(crossresut.get(1));
                    //update the initial solution list in genetic algorithm object
                    p.setArr1(is);
                    //generate random number between 0 and 10
                    int l = random.ints(0, 10).findFirst().getAsInt();
                    //if the random number is more than 8 make a mutation in the two children
                    //if (l > 8)
                      // p.mutation();

                    //sort the initial solution list based on fitness function
                    is = p.bubblesort();
                    //remove the lowest scored chromosomes from the population
                    is.remove(is.size() - 1);
                    is.remove(is.size() - 1);
                    //add the best chromosome for chart array list to use it in plot line for the algorithm performance
                    charts.add(new chart(i, p.fitness_fun(is.get(0))));
                }
                //count the groups choices respect to order of choice
                int[] f = is.get(0).clone();
                for (int i = 0; i < arr.size(); i++) {

                    if (f[i] == arr.get(i).getF())
                        fi++;
                    else if (f[i] == arr.get(i).getS())
                        sec++;
                    else if (f[i] == arr.get(i).getT())
                        th++;
                    else
                        other++;
                }


                list.clear();
                //for creating the table for comparison
                for (int i = 0; i < arr.size(); i++) {
                    list.add(new tabelView(i, arr.get(i).getGroup(), arr.get(i).getF(), arr.get(i).getS(), arr.get(i).getT(), is.get(0)[i]));
                }

                number.setCellValueFactory(new PropertyValueFactory<tabelView, Integer>("number"));
                groups.setCellValueFactory(new PropertyValueFactory<tabelView, String>("groups"));
                first.setCellValueFactory(new PropertyValueFactory<tabelView, Integer>("first"));
                second.setCellValueFactory(new PropertyValueFactory<tabelView, Integer>("second"));
                third.setCellValueFactory(new PropertyValueFactory<tabelView, Integer>("third"));
                alg.setCellValueFactory(new PropertyValueFactory<tabelView, Integer>("alg"));
                tabel.setItems(list);

                checkSubmit = true;


            }//end
            else if (txtPathFile.getText().equals("")) {


                Alert alertCreat = new Alert(Alert.AlertType.WARNING);
                alertCreat.setTitle("Warning");
                alertCreat.setHeaderText(null);
                alertCreat.setContentText("Please choose a file to read it!!");
                alertCreat.showAndWait();


            } else {


                Alert alertCreat = new Alert(Alert.AlertType.WARNING);
                alertCreat.setTitle("Warning");
                alertCreat.setHeaderText(null);
                alertCreat.setContentText("Pleas enter number of loop");
                alertCreat.showAndWait();


            }
        } catch (IllegalArgumentException exception) {


            Alert alertCreat = new Alert(Alert.AlertType.ERROR);
            alertCreat.setTitle("Error");
            alertCreat.setHeaderText(null);
            alertCreat.setContentText(exception.getMessage());
            alertCreat.showAndWait();


        }


    }


    @FXML
    void btnBrowseAction(ActionEvent event) {

        FileChooser fileChooserShares = new FileChooser();
        fileChooserShares.setTitle("Select project file ");
        fileChooserShares.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Text Files", "*.csv")
        );

        File selectedFile = fileChooserShares.showOpenDialog(null);
        if (String.valueOf(selectedFile).equals("null")) {
            return;
        } else {
            txtPathFile.setText(selectedFile.toString());
        }

    }


}
