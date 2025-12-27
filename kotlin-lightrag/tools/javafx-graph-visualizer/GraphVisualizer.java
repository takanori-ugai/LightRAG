package tools.javafx.graphvisualizer;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Point3D;
import javafx.scene.*;
import javafx.scene.control.*;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.Sphere;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.jgrapht.Graph;
import org.jgrapht.alg.connectivity.ConnectivityInspector;
import org.jgrapht.alg.drawing.FRLayoutAlgorithm2D;
import org.jgrapht.alg.drawing.model.MapLayoutModel2D;
import org.jgrapht.alg.drawing.model.Point2D;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
import org.jgrapht.io.Attribute;
import org.jgrapht.io.EdgeProvider;
import org.jgrapht.io.GraphImporter;
import org.jgrapht.io.GraphMLImporter;
import org.jgrapht.io.ImportException;
import org.jgrapht.io.VertexProvider;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.awt.geom.Rectangle2D;

/**
 * Minimal JavaFX port of tools/lightrag_visualizer/graph_visualizer.py.
 * Features:
 * - Load a GraphML file.
 * - 3D scene with spheres for nodes and cylinders for edges.
 * - Simple Fruchterman-Reingold layout (2D) mapped into 3D.
 * - Community coloring via connected components.
 * - Mouse drag to orbit, scroll to zoom, WASD/Arrow keys to pan, click node to show details.
 *
 * Dependencies (add to your build):
 * - JavaFX (controls, graphics, base, etc.)
 * - JGraphT: org.jgrapht:jgrapht-core
 */
public class GraphVisualizer extends Application {
    private static final double NODE_BASE_SIZE = 12.0;
    private static final double EDGE_RADIUS = 1.5;
    private static final double CAMERA_INITIAL_DISTANCE = -400;
    private static final double CAMERA_NEAR_CLIP = 0.1;
    private static final double CAMERA_FAR_CLIP = 10_000.0;

    private final Group root3d = new Group();
    private final Group nodeGroup = new Group();
    private final Group edgeGroup = new Group();
    private final PerspectiveCamera camera = new PerspectiveCamera(true);
    private final Map<NodeData, GraphNode> nodeLookup = new HashMap<>();
    private Graph<NodeData, DefaultEdge> graph = new SimpleGraph<>(DefaultEdge.class);

    private double anchorX;
    private double anchorY;
    private double anchorAngleX;
    private double anchorAngleY;
    private final Rotate rotateX = new Rotate(-20, Rotate.X_AXIS);
    private final Rotate rotateY = new Rotate(-20, Rotate.Y_AXIS);
    private double cameraDistance = CAMERA_INITIAL_DISTANCE;
    private GraphNode selectedNode;

    @Override
    public void start(Stage stage) {
        SubScene subScene = buildScene3d();
        BorderPane ui = buildUi(stage, subScene);

        Scene scene = new Scene(ui, 1280, 720, true);
        installCameraControls(scene);

        stage.setTitle("GraphML Visualizer (JavaFX)");
        stage.setScene(scene);
        stage.show();
    }

    private SubScene buildScene3d() {
        camera.setNearClip(CAMERA_NEAR_CLIP);
        camera.setFarClip(CAMERA_FAR_CLIP);
        camera.setTranslateZ(cameraDistance);

        root3d.getChildren().addAll(edgeGroup, nodeGroup);
        root3d.getTransforms().addAll(rotateX, rotateY);

        SubScene subScene = new SubScene(root3d, 800, 600, true, SceneAntialiasing.BALANCED);
        subScene.setFill(Color.web("#0c0c0f"));
        subScene.setCamera(camera);
        enableMouse(subScene);
        enableScroll(subScene);
        return subScene;
    }

    private BorderPane buildUi(Stage stage, SubScene subScene) {
        BorderPane root = new BorderPane();
        root.setCenter(subScene);

        Button loadButton = new Button("Load GraphML");
        loadButton.setOnAction(_ -> chooseFileAndLoad(stage));

        Button resetButton = new Button("Reset Camera");
        resetButton.setOnAction(_ -> resetCamera());

        HBox controls = new HBox(10, loadButton, resetButton, new Text("Drag: orbit | Scroll: zoom | WASD/Arrows: pan"));
        controls.setPadding(new Insets(8));

        VBox rightPane = new VBox(8);
        rightPane.setPadding(new Insets(10));
        rightPane.setPrefWidth(260);
        Label title = new Label("Node Details");
        title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");
        TextArea details = new TextArea();
        details.setEditable(false);
        details.setWrapText(true);
        VBox.setVgrow(details, Priority.ALWAYS);
        rightPane.getChildren().addAll(title, details);

        root.setTop(controls);
        root.setRight(rightPane);

        // Keep updating details panel when selection changes.
        new AnimationTimer() {
            @Override
            public void handle(long now) {
                details.setText(selectedNode == null ? "Click a node to inspect." : selectedNode.describe());
            }
        }.start();

        return root;
    }

    private void chooseFileAndLoad(Stage stage) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Open GraphML");
        chooser.getExtensionFilters().addAll(
            new FileChooser.ExtensionFilter("GraphML files", "*.graphml", "*.xml"),
            new FileChooser.ExtensionFilter("All files", "*.*")
        );
        File file = chooser.showOpenDialog(stage);
        if (file != null) {
            loadGraph(file);
        }
    }

    private void loadGraph(File file) {
        Graph<NodeData, DefaultEdge> g = new SimpleGraph<>(DefaultEdge.class);

        Map<String, NodeData> nodePool = new HashMap<>();
        VertexProvider<NodeData> vp = (id, attributes) -> nodePool.computeIfAbsent(id, key -> new NodeData(key, attributes));
        EdgeProvider<NodeData, DefaultEdge> ep = (from, to, label, attributes) -> g.getEdgeFactory().createEdge(from, to);
        GraphImporter<NodeData, DefaultEdge> importer = new GraphMLImporter<>(vp, ep);

        try (FileReader reader = new FileReader(file)) {
            importer.importGraph(g, reader);
            this.graph = g;
            layoutAndRender();
        } catch (ImportException | IOException e) {
            showAlert("Failed to load graph: " + e.getMessage());
        }
    }

    private void layoutAndRender() {
        nodeLookup.clear();
        nodeGroup.getChildren().clear();
        edgeGroup.getChildren().clear();

        if (graph.vertexSet().isEmpty()) {
            return;
        }

        Map<NodeData, Point2D> positions = computeLayout(graph);
        Map<NodeData, Color> colors = assignCommunityColors(graph);

        // Build nodes
        for (NodeData nodeData : graph.vertexSet()) {
            Point2D p = positions.getOrDefault(nodeData, new Point2D(0, 0));
            double z = ThreadLocalRandom.current().nextDouble(-50, 50);
            GraphNode node = new GraphNode(nodeData, p.getX(), p.getY(), z, colors.getOrDefault(nodeData, Color.GRAY));
            nodeLookup.put(nodeData, node);
            nodeGroup.getChildren().add(node.sphere);
        }

        // Build edges
        for (DefaultEdge e : graph.edgeSet()) {
            NodeData src = graph.getEdgeSource(e);
            NodeData tgt = graph.getEdgeTarget(e);
            GraphNode a = nodeLookup.get(src);
            GraphNode b = nodeLookup.get(tgt);
            if (a != null && b != null) {
                Cylinder cyl = buildEdgeCylinder(a.position(), b.position(), a.color);
                edgeGroup.getChildren().add(cyl);
            }
        }
    }

    private Map<NodeData, Point2D> computeLayout(Graph<NodeData, DefaultEdge> g) {
        FRLayoutAlgorithm2D<NodeData, DefaultEdge> layout = new FRLayoutAlgorithm2D<>();
        layout.setMaxIterations(400);
        MapLayoutModel2D<NodeData> model =
            new MapLayoutModel2D<>(g, new Rectangle2D.Double(-200, -200, 400, 400));
        layout.layout(g, model);
        return new HashMap<>(model.getLocations());
    }

    private Map<NodeData, Color> assignCommunityColors(Graph<NodeData, DefaultEdge> g) {
        ConnectivityInspector<NodeData, DefaultEdge> inspector = new ConnectivityInspector<>(g);
        List<Set<NodeData>> components = inspector.connectedSets();
        Map<NodeData, Color> colors = new HashMap<>();
        for (int i = 0; i < components.size(); i++) {
            Color c = Color.hsb((i * 360.0 / components.size()), 0.8, 0.95);
            for (NodeData v : components.get(i)) {
                colors.put(v, c);
            }
        }
        return colors;
    }

    private Cylinder buildEdgeCylinder(Point3D from, Point3D to, Color color) {
        Point3D diff = to.subtract(from);
        double height = diff.magnitude();
        Point3D mid = from.midpoint(to);
        Point3D axisOfRotation = diff.crossProduct(new Point3D(0, 1, 0));
        double angle = Math.acos(diff.normalize().dotProduct(new Point3D(0, 1, 0)));

        Cylinder cylinder = new Cylinder(EDGE_RADIUS, height);
        cylinder.setMaterial(new PhongMaterial(color.deriveColor(0, 1, 0.8, 0.7)));
        cylinder.getTransforms().addAll(
            new Translate(mid.getX(), mid.getY(), mid.getZ()),
            new Rotate(Math.toDegrees(angle), axisOfRotation)
        );
        return cylinder;
    }

    private void enableMouse(SubScene scene) {
        scene.setOnMousePressed(event -> {
            anchorX = event.getSceneX();
            anchorY = event.getSceneY();
            anchorAngleX = rotateX.getAngle();
            anchorAngleY = rotateY.getAngle();
            if (event.getButton() == MouseButton.PRIMARY && event.getPickResult() != null) {
                Node picked = event.getPickResult().getIntersectedNode();
                GraphNode gn = findGraphNode(picked);
                selectedNode = gn;
            }
        });

        scene.setOnMouseDragged(event -> {
            if (event.getButton() == MouseButton.SECONDARY) {
                rotateX.setAngle(anchorAngleX - (anchorY - event.getSceneY()) / 2);
                rotateY.setAngle(anchorAngleY + (anchorX - event.getSceneX()) / 2);
            }
        });
    }

    private void enableScroll(SubScene scene) {
        scene.addEventHandler(ScrollEvent.SCROLL, event -> {
            cameraDistance += event.getDeltaY() * 0.5;
            camera.setTranslateZ(cameraDistance);
        });
    }

    private void installCameraControls(Scene scene) {
        Set<KeyCode> pressed = new HashSet<>();

        scene.setOnKeyPressed(e -> pressed.add(e.getCode()));
        scene.setOnKeyReleased(e -> pressed.remove(e.getCode()));

        new AnimationTimer() {
            @Override
            public void handle(long now) {
                double step = 4.0;
                if (pressed.contains(KeyCode.W) || pressed.contains(KeyCode.UP)) {
                    root3d.setTranslateY(root3d.getTranslateY() + step);
                }
                if (pressed.contains(KeyCode.S) || pressed.contains(KeyCode.DOWN)) {
                    root3d.setTranslateY(root3d.getTranslateY() - step);
                }
                if (pressed.contains(KeyCode.A) || pressed.contains(KeyCode.LEFT)) {
                    root3d.setTranslateX(root3d.getTranslateX() + step);
                }
                if (pressed.contains(KeyCode.D) || pressed.contains(KeyCode.RIGHT)) {
                    root3d.setTranslateX(root3d.getTranslateX() - step);
                }
            }
        }.start();
    }

    private void resetCamera() {
        rotateX.setAngle(-20);
        rotateY.setAngle(-20);
        cameraDistance = CAMERA_INITIAL_DISTANCE;
        camera.setTranslateZ(cameraDistance);
        root3d.setTranslateX(0);
        root3d.setTranslateY(0);
    }

    private GraphNode findGraphNode(Node picked) {
        if (picked == null) return null;
        for (GraphNode gn : nodeLookup.values()) {
            if (gn.sphere == picked || picked.getParent() == gn.sphere) {
                return gn;
            }
        }
        return null;
    }

    private void showAlert(String msg) {
        Alert alert = new Alert(Alert.AlertType.ERROR, msg, ButtonType.OK);
        alert.showAndWait();
    }

    private double randomRange(double min, double max) {
        return ThreadLocalRandom.current().nextDouble(min, max);
    }

    private static class GraphNode {
        private final NodeData data;
        private final Sphere sphere;
        private final Color color;

        GraphNode(NodeData data, double x, double y, double z, Color color) {
            this.data = data;
            this.color = color;
            this.sphere = new Sphere(NODE_BASE_SIZE);
            this.sphere.setMaterial(new PhongMaterial(color));
            this.sphere.setTranslateX(x);
            this.sphere.setTranslateY(y);
            this.sphere.setTranslateZ(z);
        }

        Point3D position() {
            return new Point3D(sphere.getTranslateX(), sphere.getTranslateY(), sphere.getTranslateZ());
        }

        String describe() {
            StringBuilder sb = new StringBuilder();
            sb.append("Node: ").append(data.id).append("\n");
            if (!data.attributes.isEmpty()) {
                sb.append("\nAttributes:\n");
                data.attributes.forEach((k, v) -> sb.append("- ").append(k).append(": ").append(v).append("\n"));
            }
            return sb.toString();
        }
    }

    private static class NodeData {
        private final String id;
        private final Map<String, String> attributes;

        NodeData(String id, Map<String, Attribute> attrs) {
            this.id = id;
            this.attributes =
                (attrs == null ? Collections.<String, Attribute>emptyMap() : attrs).entrySet().stream().collect(
                    Collectors.toMap(Map.Entry::getKey, e -> e.getValue().getValue()));
        }

        @Override
        public String toString() {
            return id;
        }

        @Override
        public int hashCode() {
            return Objects.hash(id);
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof NodeData other)) return false;
            return Objects.equals(id, other.id);
        }
    }

    // Simple wrapper to allow Translate transform with Point3D.
    private static class Translate extends javafx.scene.transform.Translate {
        Translate(double x, double y, double z) {
            super(x, y, z);
        }
    }

    private static class Rotate extends javafx.scene.transform.Rotate {
        Rotate(double angle, Point3D axis) {
            super(angle, axis);
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
