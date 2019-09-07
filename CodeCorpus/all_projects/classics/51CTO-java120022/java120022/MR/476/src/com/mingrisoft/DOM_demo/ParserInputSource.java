/**
 *second 
 */
package com.mingrisoft.DOM_demo;

import java.io.IOException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

/**
 * @author baiweiming
 * 
 */
public class ParserInputSource {

	/**
	 * 使用InputSource读取XML文件
	 * 
	 * @param path
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	public void parseInputSource(InputSource is)
			throws ParserConfigurationException, SAXException, IOException {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = documentBuilderFactory
				.newDocumentBuilder();
		dombuilder.parse(is);

	}

	public static void main(String[] arg) {
		ParserInputSource parserFile = new ParserInputSource();
		String systemId = "xmldemo/books.xml";
		InputSource is = new InputSource(systemId);
		try {
			parserFile.parseInputSource(is);
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
