package sample;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.AnchorPane;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;

public class controlerchart  implements Initializable {



    @FXML
    private AnchorPane chartplot;

    @FXML
    private Label label;

    @FXML
    private LineChart<?, ?> ch;

    @FXML
    private Label first;

    @FXML
    private Label iteration;



    @FXML
    private Label other;

    @FXML
    private Label second;

    @FXML
    private Label third;

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        XYChart.Series series = new XYChart.Series();
        series.setName("Score of the chromosome");
        ArrayList<chart>c = Controller.charts ;
            first.setText("First : "+Integer.toString(Controller.fi));

            second.setText("Second : "+Integer.toString(Controller.sec));
            third.setText("Third : " +Integer.toString(Controller.th));
            other.setText("Other : " +Integer.toString(Controller.other));

        for (int i = 0 ; i < c.size() ; i++)
            series.getData().add(new XYChart.Data(Integer.toString(c.get(i).getX()) ,c.get(i).getY())) ;


        ch.getData().add(series) ;


    }
}
