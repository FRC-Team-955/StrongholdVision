//http://stackoverflow.com/questions/12975341/to-string-is-not-a-member-of-std-says-so-g
//Makes it so that you can convert numbers to strings because that is a bug in G++
#include <string>
#include <sstream>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str();
    }
}

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <pthread.h>
#include <thread>
#include <arpa/inet.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pugixml.hpp>
#include <regex>
#include <iterator>
#include <math.h>
#include "ntcore.h"
#include "networktables/NetworkTable.h"
#include <netinet/in.h>
#include <netdb.h>


using namespace cv;
using namespace pugi;

bool UI = false;

//Images
Mat src, hsv_src, hsv_filtered, masked_bgr, rect_ROI, rect_ROI_canny;
//Trackbars
const int slider_max_sat_val = 256;
const int slider_max_hue = 179;

//Data

//Reflective tape (High Light)
//Lower
int hue_slider_lower = 49;
int sat_slider_lower = 0;
int val_slider_lower = 223;
//Upper
int hue_slider_upper = 97;
int sat_slider_upper = 164;
int val_slider_upper = 256;
//Canny
int canny_slider = 35;
//Hough
int hough_slider = 256;
//Morph
int morph_slider = 2;
int morph_slider_max = 8;
//Shape-detect
int shape_slider_max = 100;
int shape_slider_lower = 20;
int shape_slider_upper = 45;


/*
//Reflective tape (Low Light)
//Lower
int hue_slider_lower = 81;
int sat_slider_lower = 29;
int val_slider_lower = 235;
//Upper
int hue_slider_upper = 94;
int sat_slider_upper = 256;
int val_slider_upper = 256;
//Canny
int canny_slider = 69;
//Hough
int hough_slider = 256;
//Morph
int morph_slider = 2;
int morph_slider_max = 8;
*/

/*
//Blue Tape
//Lower
int hue_slider_lower = 102;
int sat_slider_lower = 90;
int val_slider_lower = 91;
//Upper
int hue_slider_upper = 115;
int sat_slider_upper = 220;
int val_slider_upper = 256;
*/

struct LineInfo {
    Vec4i line;
    float length;
    float slope;
    Point midPt;
};

//http://stackoverflow.com/questions/9074202/opencv-2-centroid
cv::Point computeCentroid(const cv::Mat &mask) {
    cv::Moments m = moments(mask, true);
    cv::Point center(m.m10/m.m00, m.m01/m.m00);
    return center;
}

int LineCompareLength (const void * a, const void * b)
{
    LineInfo* A_cast = (LineInfo*) a ;
    LineInfo* B_cast = (LineInfo*) b ;

  if ( *(float*)&A_cast->length <  *(float*)&B_cast->length ) return 1;
  if ( *(float*)&A_cast->length == *(float*)&B_cast->length ) return 0;
  if ( *(float*)&A_cast->length >  *(float*)&B_cast->length ) return -1;
}

int compareDoubles (const void * a, const void * b) {
    if ( *(double*)&a <  *(double*)&b ) return -1;
    if ( *(double*)&a == *(double*)&b ) return 0;
    if ( *(double*)&a >  *(double*)&b ) return 1;
}

float calcSlope (Vec4i line) {
    if ((line[2] - line [0]) != 0) {
        return ((float)line[3] - (float)line [1]) / ((float)line[2] - (float)line [0]);
    } else {
        return 9999999; //Pseudo infinity
    }
}

Point calcMidPt(Vec4i line) {
    return Point((line[0] + line[2]) / 2, (line[1] + line[3]) / 2);
}

float calcLength(Vec4i line) {
    int distancex = (line[2] - line[0]) * (line[2] - line[0]);
    int distancey = (line[3] - line[1]) * (line[3] - line[1]);
    return sqrt((float)distancex + (float)distancey);
}

LineInfo GetInfo(Vec4i line) {
    LineInfo result;
    result.line = line;
    result.length = calcLength(line);
    result.slope = calcSlope(line);
    result.midPt = calcMidPt(line);
    return result;
}

void on_trackbar (int, void*) {
    xml_document settings;
    xml_node sliders = settings.append_child("sliders");
    sliders.append_attribute("HueLw").set_value(hue_slider_lower);
    sliders.append_attribute("HueUp").set_value(hue_slider_upper);
    sliders.append_attribute("SatLw").set_value(sat_slider_lower);
    sliders.append_attribute("SatUp").set_value(sat_slider_upper);
    sliders.append_attribute("ValLw").set_value(val_slider_lower);
    sliders.append_attribute("ValUp").set_value(val_slider_upper);
    sliders.append_attribute("Canny").set_value(canny_slider);
    sliders.append_attribute("Hough").set_value(hough_slider);
    sliders.append_attribute("Morph").set_value(morph_slider);
    sliders.append_attribute("ShapeLw").set_value(shape_slider_lower);
    sliders.append_attribute("ShapeUp").set_value(shape_slider_upper);
    settings.save_file("/home/ubuntu/slidersettings.xml");
}

void create_trackbars () {
    Mat useless(1, 1920 / 3, 0);
    imshow("Sliders", useless);

    //Hue
    createTrackbar("Hue Lw", "Sliders", &hue_slider_lower, slider_max_hue, on_trackbar);
    createTrackbar("Hue Up", "Sliders", &hue_slider_upper, slider_max_hue, on_trackbar);

    //Sat
    createTrackbar("Sat Lw", "Sliders", &sat_slider_lower, slider_max_sat_val, on_trackbar);
    createTrackbar("Sat Up", "Sliders", &sat_slider_upper, slider_max_sat_val, on_trackbar);

    //Val
    createTrackbar("Val Lw", "Sliders", &val_slider_lower, slider_max_sat_val, on_trackbar);
    createTrackbar("Val Up", "Sliders", &val_slider_upper, slider_max_sat_val, on_trackbar);

    //Canny
    createTrackbar("Canny", "Sliders", &canny_slider, slider_max_sat_val, on_trackbar);

    //Hough
    createTrackbar("Hough", "Sliders", &hough_slider, slider_max_sat_val, on_trackbar);

    //Morph
    createTrackbar("Morph", "Sliders", &morph_slider, morph_slider_max, on_trackbar);

    //Goal Shape-detect
    createTrackbar("ShapeLw", "Sliders", &shape_slider_lower, shape_slider_max, on_trackbar);
    createTrackbar("ShapeUp", "Sliders", &shape_slider_upper, shape_slider_max, on_trackbar);
}

pthread_mutex_t mutex;
const char* socket_send_xml_buffer;

char buf[2048];

void* netcode_thread (void* arg) {
    std::cout << "Netcode booting..." << std::endl;

    /*
    auto nt = NetworkTable::GetTable("test");

    //nt->SetClientMode();
    nt->SetServerMode();
    //nt->SetIPAddress("nroborio-955-frc.local\n");

    nt->Initialize();
    std::this_thread::sleep_for(std::chrono::seconds(5));

    //if (nt->IsConnected) {
        std::cout << "Network table created, connected" << std::endl;/*
    } else {
        std::cout << "Network table connect failed" << std::endl;
    }

    for (;;) {
        pthread_mutex_lock(&mutex);
        memcpy(buf, socket_send_xml_buffer, strlen((socket_send_xml_buffer)));
        pthread_mutex_unlock(&mutex);
        nt->PutString("value", buf);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    */

    /*
    struct sockaddr_in server_addr,client_addr;
    socklen_t clientlen = sizeof(client_addr);
    int option, port, reuse;
    int server, client;
    size_t buflen;
    size_t nread;

    port = 5805;

    // setup socket address structure
    memset(&server_addr,0,sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = INADDR_ANY;

      // create socket
    server = socket(PF_INET,SOCK_STREAM,0);
    if (!server) {
        perror("socket");
        exit(-1);
    }

      // set socket to immediately reuse port when the application closes
    reuse = 1;
    if (setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("setsockopt");
        exit(-1);
    }

      // call bind to associate the socket with our local address and
      // port
    if (bind(server,(const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(-1);
    }

      // convert the socket to listen for incoming connectionŘs
    if (listen(server,SOMAXCONN) < 0) {
        perror("listen");
        exit(-1);
    }

    //printf("Finished binding");
    std::cout << "Finish binding netcode\n" << std::endl;

    // allocate buffer
    buflen = 1024;
    buf = new char[buflen+1];

    // accept clients
    while ((client = accept(server,(struct sockaddr *)&client_addr,&clientlen)) > 0) {
        std::cout << "Client accepted\n" << std::endl;

        // loop to handle all requests
        while (1) {
            //std::cout << "Printed... " << std::endl;


            // read a request
            memset(buf,0,buflen);
            nread = recv(client,buf,buflen,0);

            if (nread > 0) {
                std::cout << "Got Request: " << buf << std::endl;
            }


            if (nread == 0)
                break;

            //size_t newstr_length = strlen(newstr);

            //printf("newstr_length = %d\n",(int)newstr_length) ;

            // send a response
            pthread_mutex_lock(&mutex);
            memcpy(buf, socket_send_xml_buffer, strlen((socket_send_xml_buffer)));
            pthread_mutex_unlock(&mutex);
            send(client, socket_send_xml_buffer, strlen(socket_send_xml_buffer), 0);


             //memset(buf,0,buflen);
        }

        std::cout << "Client disconnected\n" << std::endl;
        close(client);
    }

    std::cout << "Server has died" << std::endl;

    close(server);
    */

    //http://www.cs.rpi.edu/~moorthy/Courses/os98/Pgms/client.c

    struct sockaddr_in serv_addr;
    struct hostent *server;
    int sockfd, portno, n;
    portno = 5805;

    server = gethostbyname("roborio-955-frc.local");//gethostbyname("roborio-955-frc.local");
    if (server == NULL) {
        std::cout << "No such host" << std::endl;
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);

    for (;;) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            std::cout << "Socket could not open" << std::endl;
            break;
        }

        if (connect(sockfd,(struct sockaddr *)&serv_addr,sizeof(serv_addr)) < 0) {
            std::cout << "Connect error" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        std::cout << "Connected to RIO\n" << std::endl;

        // loop to handle all requests
        while (1) {
            //std::cout << "Printed... " << std::endl;


            // read a request
            //memset(buf,0,strlen(buf));

            ssize_t nread;
            nread = recv(sockfd,buf,sizeof(buf), 0);

            if (nread > 0) {
                //printf("%d\n", nread);
                buf[nread] = 0;
                std::cout << "Got Request: " << buf << std::endl;
            } else {
                break;
            }

            // send a response
            pthread_mutex_lock(&mutex);
            size_t buflen = strlen(socket_send_xml_buffer);
            memcpy(buf, socket_send_xml_buffer, buflen);
            pthread_mutex_unlock(&mutex);
            send(sockfd, buf, buflen, 0);
            buf[buflen] = 0;
            std::cout << "Sent: " << buf << std::endl;



             //memset(buf,0,buflen);
        }

        std::cout << "Disconnected\n" << std::endl;
        close(sockfd);


    }

    std::cout << "Client loop failed\n" << std::endl;


    return NULL;
}


int main(int numArgs, char** args)
{
    UI = (numArgs == 2); //Set in UI mode if there is an argument

    std::cout << "Vision Starting..." << std::endl;
    xml_document settings;
    try {
        xml_parse_result result = settings.load_file("/home/ubuntu/slidersettings.xml");
        xml_node settingRoot = settings.child("sliders");
        hue_slider_lower = settingRoot.attribute("HueLw").as_int();
        hue_slider_upper = settingRoot.attribute("HueUp").as_int();
        sat_slider_lower = settingRoot.attribute("SatLw").as_int();
        sat_slider_upper = settingRoot.attribute("SatUp").as_int();
        val_slider_lower = settingRoot.attribute("ValLw").as_int();
        val_slider_upper = settingRoot.attribute("ValUp").as_int();
        canny_slider = settingRoot.attribute("Canny").as_int();
        hough_slider = settingRoot.attribute("Hough").as_int();
        morph_slider = settingRoot.attribute("Morph").as_int();
        shape_slider_lower = settingRoot.attribute("ShapeLw").as_int();
        shape_slider_upper = settingRoot.attribute("ShapeUp").as_int();
    } catch (int e) {
        std::cout << "Slider XML parse failed" << std::endl;
    }



    //TODO: Add arg to disable gui (-nogui)
    pthread_t other;


	VideoCapture cap(-1); // open the default camera
	if(!cap.isOpened()) {  // check if we succeeded
        std::cout << "Camera not found" << std::endl;
		return -1;
    }

    const int pixelWidth = 1920 / 2;
    const int pixelHeight = 1080 / 2;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, pixelWidth);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, pixelHeight);
	//double temp = cap.get(CV_CAP_PROP_EXPOSURE);
	//cap.set(CV_CAP_PROP_EXPOSURE, -40.0);
	//std::cout << temp << std::endl;

	if (UI) namedWindow("Output", CV_WINDOW_AUTOSIZE);
    vector< vector <Point> > contours;
    if (UI) create_trackbars();

    u_int frameCount = 0;

    std::cout << "Starting netcode thread..." << std::endl;

    // Initialize the mutex
    if(pthread_mutex_init(&mutex, NULL))
    {
        std::cout << "Unable to initialize a mutex\n" << std::endl;
        return -1;
    }

    if(pthread_create(&other, NULL, &netcode_thread, NULL))
    {
        std::cout << "Unable to spawn thread\n" << std::endl;
        return -1;
    }

    std::cout << "Start vision loop" << std::endl;

	for(;;)
	{
        if (waitKey(1) == 27) break;

        frameCount++;
		cap >> src; // get a new frame from camera
		if (src.empty())
			break;

        //imshow("Source", src);

        cvtColor(src, hsv_src, COLOR_BGR2HSV); //Convert to HSV colorspace
        inRange(hsv_src, Scalar(hue_slider_lower, sat_slider_lower, val_slider_lower), Scalar(hue_slider_upper, sat_slider_upper, val_slider_upper), hsv_filtered); //HSV Threshold
        masked_bgr.setTo(Scalar(0,0,0,0)); //Erase old data from last render (background to the HSV threshhold)

        Mat element = getStructuringElement(MORPH_RECT, Size( 2*morph_slider + 1, 2*morph_slider+1 ), Point( morph_slider, morph_slider ) ); //Make sure that objects have a certain area
        morphologyEx(hsv_filtered, hsv_filtered, MORPH_OPEN, element);
        src.copyTo(masked_bgr, hsv_filtered); //Mask the final image by the HSV selection

        Mat contour_hsv_filtered_copy = hsv_filtered.clone();
        findContours(contour_hsv_filtered_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

        vector< vector <Point> > hull(contours.size());

        xml_document send_to_RIO;

        xml_node visionNode = send_to_RIO.append_child("Vision");
        visionNode.append_attribute("frameNumber").set_value(frameCount);

        u_int shapeCount = 0;

        for (int idx = 0; idx < contours.size(); idx++)
        {
            Rect boundRect = boundingRect(contours[idx]); // boundingbox
            Point boundCenter = Point(boundRect.width / 2, boundRect.height / 2) + boundRect.tl();

            convexHull(Mat(contours[idx]), hull[0], false);
            //drawContours(masked_bgr, boundRect, idx, Scalar(0, 0, 255));
            //drawContours(masked_bgr, contours, idx, Scalar(0, 255, 255), 1, 8, noArray(), 0, Point());

            //Find the U shape
            bool isTarget = true;
            double hullArea = contourArea(hull[0]);
            double contArea = contourArea(contours[idx]);
            //double perimeter = arcLength(contours[idx], false);
            double areaRatio = hullArea / contArea;
            //std::cout << patch::to_string(areaRatio) << std::endl;

            if (/*2.0*/((float)shape_slider_lower / 10.0f) < areaRatio && areaRatio < ((shape_slider_upper) / 10.0f)/*4.5*/) {
                shapeCount++;
                rectangle(masked_bgr, boundRect, Scalar(0, 255, 255), 2);
                rect_ROI = hsv_filtered(boundRect);
                if (canny_slider == 0) { canny_slider = 1; }
                Canny(rect_ROI, rect_ROI_canny, canny_slider, 0, 3);

                //imshow("Canny", rect_ROI_canny);

                vector<Vec4i> lines;
                HoughLinesP(rect_ROI_canny, lines, 1, CV_PI/180, canny_slider, canny_slider, 10);

                LineInfo* lineInforCacheprt = new LineInfo[lines.size()];
                LineInfo line_exclude = GetInfo(Vec4i(0,0,0,0));
                for( size_t i = 0; i < lines.size(); i++ )
                {
                    LineInfo current = GetInfo(lines[i]);
                    if (current.slope < 1.0 && current.slope > -1.0) { //Exclude slopes of >1.0 (45 degrees)
                        lineInforCacheprt[i] = current;
                    } else {
                        lineInforCacheprt[i] = line_exclude;
                    }
                }

                qsort(lineInforCacheprt,
                        lines.size(),
                        sizeof(LineInfo),
                        LineCompareLength
                        );

                //size_t i = 0;
                double transMag = 0.0;
                for( size_t i = 0; i < lines.size(); i++ )
                {
                    Vec4i l = lines[i];
                    if (calcLength(l) == lineInforCacheprt[0].length) {
                        line(masked_bgr, Point(l[0], l[1]) + boundRect.tl(), Point(l[2], l[3]) + boundRect.tl(), Scalar(255,0,255), 2, CV_AA);
                        double degrees = (atan2((double)l[3] - (double)l[1], (double)l[2] - (double)l[0]) * 180.0) / CV_PI;
                        transMag = degrees * (100.0 / 45.0);
                        putText(masked_bgr, ("Translate: " + patch::to_string((int)transMag)).c_str(), boundCenter + Point(-65, 105), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
                    }
                }


                circle(masked_bgr, boundCenter, 5, Scalar(0, 255, 0), 1, 8, 0);
                //double rotateMag = (((double)boundCenter.x / (double)pixelWidth) - 0.5) * 200.0;
                int screenCenterX = (pixelWidth / 2);
                int distanceToCenterX = screenCenterX - (boundCenter.x + (boundRect.width / 2));
                int distanceToCenterXMagnitude = ((float)distanceToCenterX / screenCenterX) * 160.0f;
                int yPos = boundCenter.y;
                double distanceToGoal = (3e-06 * pow(yPos, 3)) + (-0.001 * pow(yPos, 2))  + (0.315 * yPos) - 0.557;
                if (distanceToCenterXMagnitude > 100.0) {distanceToCenterXMagnitude = 100.0;}
                if (distanceToCenterXMagnitude < -100.0) {distanceToCenterXMagnitude = -100.0;}
                putText(masked_bgr, ("Rotate: " + patch::to_string(distanceToCenterXMagnitude)).c_str(), boundCenter + Point(-65, 85), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
                putText(masked_bgr, ("Distance: " + patch::to_string(distanceToGoal)).c_str(), boundCenter + Point(-65, 125), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
                //putText(masked_bgr, ("X: " + patch::to_string(boundCenter.x) + " Y: " + patch::to_string(boundCenter.y)).c_str(), boundCenter + Point(-65, 85), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));

                xml_node subNode;

                //TODO: Sort the goals by X position on the screen

                //if (shapeCount == 0) {
                    subNode = visionNode.append_child("goal");

                    //visionNode.append_child("FrameInfo").append_attribute("FrameCount").set_value(frameCount);
                    subNode.append_attribute("translation").set_value((int)transMag);
                    subNode.append_attribute("rotation").set_value(distanceToCenterXMagnitude);
                    subNode.append_attribute("distance").set_value(distanceToGoal);
                    //subNode.append_attribute("area").set_value(boundRect.width * boundRect.height);
                    subNode.append_attribute("area").set_value(contArea);

                //}

                //xml_node distance_node = subNode.append_child("Distance");
                //distance_node.set_value(patch::to_string(boundRect.height).c_str());

                delete(lineInforCacheprt);

            } else {
                //std::cout << patch::to_string(areaRatio) << std::endl;
            }






        }

		if (UI) imshow("Output", masked_bgr);

		if((char)waitKey(10) == 27) break;

        std::stringstream ss;
        send_to_RIO.save(ss);
        std::string tmp = ss.str();
        //std::replace(tmpDest.begin(), tmpDest.end(), "\n", " ");
        const char* cstr = tmp.c_str();
        size_t stringLength = strlen(cstr);
        char* toRep = new char[stringLength+2];
        strcpy(toRep, cstr);
        for (u_int i = 0; i < stringLength; i++) {
            if (cstr[i] == '\n') {
                toRep[i] = ' ';
            } else {
                toRep[i] = cstr[i];
            }
        }
        toRep[stringLength] = '\n';
        toRep[stringLength+1] = '\0';


		//TODO send XML
		pthread_mutex_lock(&mutex);
		socket_send_xml_buffer = toRep;
		pthread_mutex_unlock(&mutex);

        //delete(cstr);


	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	std::cout << "Vision crashed" << std::endl;

	return 0;
}
