package net.sansa_stack.integration.test;

import com.github.dockerjava.api.model.ContainerNetwork;
import com.google.common.io.ByteSource;
import org.aksw.commons.util.exception.ExceptionUtilsAksw;
import org.aksw.commons.util.healthcheck.HealthcheckRunner;
import org.aksw.jena_sparql_api.core.utils.QueryExecutionUtils;
import org.aksw.jena_sparql_api.delay.extra.Delayer;
import org.aksw.jena_sparql_api.delay.extra.DelayerDefault;
import org.aksw.jena_sparql_api.rx.SparqlRx;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.jena.atlas.test.Gen;
import org.apache.jena.query.*;
import org.apache.jena.rdfconnection.RDFConnection;
import org.apache.jena.rdfconnection.RDFConnectionFactory;
import org.apache.jena.riot.system.stream.StreamManager;
import org.apache.jena.sparql.engine.http.QueryEngineHTTP;
import org.apache.spark.SparkContext;
import org.apache.spark.deploy.SparkSubmit;
import org.apache.spark.sql.SparkSession;
import org.junit.*;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testcontainers.Testcontainers;
import org.testcontainers.containers.ContainerState;
import org.testcontainers.containers.DockerComposeContainer;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.Network;
import org.testcontainers.containers.output.Slf4jLogConsumer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.containers.wait.strategy.WaitStrategy;
import org.testcontainers.images.builder.ImageFromDockerfile;
import org.testcontainers.utility.DockerImageName;
import org.testcontainers.utility.MountableFile;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class SansaIT {
    private static final Logger logger = LoggerFactory.getLogger(SansaIT.class);

    protected Map<String, String> env;

    protected Path bundleFolder;
    protected DockerComposeContainer environment;
    // protected ContainerNetwork network;

    /* The network ID of the docker-composed-based spark setup */
    protected String networkId;

    protected Path stackJarBundleFolder;
    protected Path exampleJarBundleFolder;

    //protected String sparkMasterUrl;

    protected String sparkMasterHost;

    protected int sparkMasterPort;
    protected int sparkMasterWebUiPort;
    protected int sparkTestPort; // sansaTestPort?

    protected String sparkMasterUrl;

    /**
     * The name of an environment variable that points to the folder where to look for the jar bundle
     */
    public String STACK_JAR_BUNDLE_FOLDER_KEY = "STACK_JAR_BUNDLE_FOLDER";
    public String EXAMPLE_JAR_BUNDLE_FOLDER_KEY = "EXAMPLE_JAR_BUNDLE_FOLDER";

    public String SPARK_MASTER_WEBUI_PORT_KEY = "SPARK_MASTER_WEBUI_PORT";
    public String SPARK_MASTER_PORT_KEY = "SPARK_MASTER_PORT";

    /** Some deployed spark apps open a port where they provide their service
     * (at present we expect only 1 additional port to be opened) */
    public String SPARK_TEST_PORT_KEY = "SPARK_TEST_PORT";


    public static final Query spoQuery = QueryFactory.create("SELECT * { ?s ?p ?o }");


    public GenericContainer sparkSubmit(Path jarBundlePath, String[] cmdArgs) {
        String basePath = "/spark/bin/";
        String[] cmd = ArrayUtils.insert(0, cmdArgs, basePath + "spark-submit");

        ImageFromDockerfile sparkSubmitImage = new ImageFromDockerfile()
            .withDockerfileFromBuilder(builder -> builder
                .from("bde2020/spark-base:3.0.1-hadoop3.2")
                .expose(sparkTestPort)
                .build());

        // DockerImageName.parse("bde2020/spark-master:3.0.1-hadoop3.2")

        System.out.println("Cmd: " + Arrays.toString(cmd));
        GenericContainer result = new GenericContainer(sparkSubmitImage)
                .withCopyFileToContainer(MountableFile.forHostPath(jarBundlePath), "/spark/bin/" + jarBundlePath.getFileName())
                .withNetwork(newNetwork(networkId))
                .withCommand(cmd);

        boolean debug = false;

        if (debug) {
            // Log output from the containers
            result.withLogConsumer(new Slf4jLogConsumer(logger));
        }

        return result;
    }


    @BeforeClass
    public static void beforeClass() throws IOException {
        // System.out.println(SansaIntegrationTests.class.getClassLoader().getResource("rdf.nt"));
        // throw new RuntimeException("bail out");
        // Files.createDirectories(Paths.get(StandardSystemProperty.JAVA_IO_TMPDIR.value()).resolve("spark-events"));
    }

    @Before
    public void before() {

        env = System.getenv();
        stackJarBundleFolder = Paths.get(env.getOrDefault(STACK_JAR_BUNDLE_FOLDER_KEY, "../sansa-stack/sansa-stack-spark/target/"));
        exampleJarBundleFolder = Paths.get(env.getOrDefault(EXAMPLE_JAR_BUNDLE_FOLDER_KEY, "../sansa-examples/sansa-examples-spark/target/"));


        sparkMasterWebUiPort = Integer.parseInt(env.getOrDefault(SPARK_MASTER_WEBUI_PORT_KEY, "7541"));
        sparkMasterPort = Integer.parseInt(env.getOrDefault(SPARK_MASTER_PORT_KEY, "7542"));
        sparkTestPort = Integer.parseInt(env.getOrDefault(SPARK_TEST_PORT_KEY, "7549"));

        // Testcontainers.exposeHostPorts(sparkMasterPort);
        // Testcontainers.exposeHostPorts(sparkTestPort);

        // sparkMasterUrl = "spark://host.testcontainers.internal:" + sparkMasterPort;

        sparkMasterUrl = "spark://spark-master:7077";
        // sparkMasterUrl = "spark://spark-master:" + sparkTestPort;
        //sparkMasterHost = "localhost";
        //sparkMasterUrl = "spark://" + sparkMasterHost + ":" + "7077";//sparkMasterPort;


        environment =
                new DockerComposeContainer(new File("src/test/resources/docker-compose.yml"))
                        .withExposedService("spark-master", 7077)
                        .withExposedService("spark-master", 8080);
//                        .withExposedService("spark-master", sparkMasterWebUiPort)
//                        .withExposedService("spark-master", sparkMasterPort);

        environment.start();

        ContainerState state = (ContainerState)environment.getContainerByServiceName("spark-master_1").orElse(null);
//        System.out.println("CONTAINER: " + state);
        networkId = state.getContainerInfo().getNetworkSettings().getNetworks().entrySet().iterator().next().getValue().getNetworkID();
//        System.out.println("NETWORK: " + network);

        System.out.println("NETWORKS:");
        state.getContainerInfo().getNetworkSettings().getNetworks().entrySet().forEach(e -> System.out.println("NETWORK: " + e));
    }


    @After
    public void after() {
        environment.stop();
//        SparkSession sc = SparkSession.getActiveSession().getOrElse(null);
//        if (sc != null) {
//            sc.stop();
//        }
//        sc = SparkSession.getDefaultSession().getOrElse(null);
//        if (sc != null) {
//            sc.stop();
//        }
    }

    // @Test
    public void test() throws Exception {

        System.out.println("Started");
        ByteSource bs = new ByteSource() {
            @Override
            public InputStream openStream() throws IOException {
                return new URL("http://localhost:8080").openStream();
            }
        };

        System.out.println("READ: " + bs.asCharSource(StandardCharsets.UTF_8).read());
    }

    public void sparkSubmitAndValidateBindingCounts(
            Path jarBundleHostPath,
            String[] args,
            String sparqlEndpointUrl,
            Map<Query, Long> queryAndBindingCount) {

        try (GenericContainer submitContainer = sparkSubmit(jarBundleHostPath, args)
                .withExposedPorts(sparkTestPort)
                .waitingFor(Wait.forLogMessage(".*", 1))) {

            submitContainer.setPortBindings(Arrays.asList("" + sparkTestPort + ":" + sparkTestPort));
            submitContainer.start();

            for (Map.Entry<Query, Long> e : queryAndBindingCount.entrySet()) {
                Query query = e.getKey();
                long expectedBindingCount = e.getValue();

                long resultSetSize = HealthcheckRunner.builder()
                        .setRetryCount(100)
                        .setInterval(1, TimeUnit.SECONDS)
                        .setAction(() -> SparqlRx.fetchBindingCount(sparqlEndpointUrl, query).blockingGet())
                        .addContinuationCondition(submitContainer::isRunning)
                        .build()
                        .call();

                Assert.assertEquals(expectedBindingCount, resultSetSize);
            }
        }

    }

    @Test
    public void testSparqlifySubmit() throws Exception {
        Path jarBundleHostPath = IOUtils.findLatestFile(exampleJarBundleFolder, "*jar-with-dependencies*").toAbsolutePath().normalize(); //.toString();
        Path jarBundleContainerPath = jarBundleHostPath.getFileName();
        String sparqlEndpointUrl = "http://localhost:" + sparkTestPort + "/sparql";

        String[] args = new String[] {
                 "--class", "net.sansa_stack.examples.spark.query.SPARQLEngineExample",
                "--master", sparkMasterUrl,
                "--num-executors", "2",
                "--executor-memory", "1G",
                "--executor-cores", "2",
                "/spark/bin/" + jarBundleContainerPath.toString(),
                "endpoint",
                "--sparql-engine", "Sparqlify",
                "-i", "rdf.nt",
                "-p", Integer.toString(sparkTestPort)
        };

        sparkSubmitAndValidateBindingCounts(
                jarBundleHostPath, args, sparqlEndpointUrl,
                Collections.singletonMap(spoQuery, 106l));
    }

    @Test
    public void testOntopSubmit() throws Exception {
        Path jarBundleHostPath = IOUtils.findLatestFile(exampleJarBundleFolder, "*jar-with-dependencies*").toAbsolutePath().normalize(); //.toString();
        Path jarBundleContainerPath = jarBundleHostPath.getFileName();
        String sparqlEndpointUrl = "http://localhost:" + sparkTestPort + "/sparql";

        String[] args = new String[] {
                // "--class", "net.sansa_stack.examples.spark.query.Sparklify",
                "--class", "net.sansa_stack.examples.spark.query.SPARQLEngineExample",
                "--master", sparkMasterUrl,
                "--num-executors", "2",
                "--executor-memory", "1G",
                "--executor-cores", "2",
                "/spark/bin/" + jarBundleContainerPath.toString(),
                "endpoint",
                "--sparql-engine", "Ontop",
                "-i", "rdf.nt",
                "-p", Integer.toString(sparkTestPort)
        };

        sparkSubmitAndValidateBindingCounts(
                jarBundleHostPath, args, sparqlEndpointUrl,
                Collections.singletonMap(spoQuery, 106l));
    }


    // Workaround based on https://github.com/testcontainers/testcontainers-java/issues/856
    public static Network newNetwork(String id) {
        return new Network() {
            @Override
            public String getId() {
                return id;
            }

            @Override
            public void close() {
            }

            @Override
            public Statement apply(Statement base, Description description) {
                return null;
            }
        };
    }
}
