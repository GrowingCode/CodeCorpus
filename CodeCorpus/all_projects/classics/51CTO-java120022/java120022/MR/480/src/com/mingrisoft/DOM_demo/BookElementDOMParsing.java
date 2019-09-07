package com.mingrisoft.DOM_demo;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.w3c.dom.DOMException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;

import com.mingrisoft.entity.BookElement;

/**
 * @author bwm
 * 
 */
public class BookElementDOMParsing {

	private List<BookElement> bookList = new ArrayList<BookElement>();

	private BookElement bookElement;

	/**
	 * 读取XML文件
	 * 
	 * @param path
	 * @return
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	public Document parseReadFile(String path) {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = null;
		try {
			dombuilder = documentBuilderFactory.newDocumentBuilder();
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		File file = new File(path);
		try {
			return dombuilder.parse(file);
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * 获取{@link Node}的name和textcontent
	 * 
	 * @param document
	 */
	public List<BookElement> getBook(Element element) {
		NodeList list = element.getElementsByTagName("book:book");

		for (int i = 0; i < list.getLength(); i++) {
			bookElement = new BookElement();
			NodeList name = element.getElementsByTagName("book:name");
			NodeList publisher = element.getElementsByTagName("book:publisher");
			NodeList company = element.getElementsByTagName("book:company");
			NodeList author = element.getElementsByTagName("book:author");
			NodeList ISBN = element.getElementsByTagName("book:ISBN");
			NodeList price = element.getElementsByTagName("book:price");
			NodeList url = element.getElementsByTagName("book:url");
			bookElement.setName(name.item(i).getTextContent());
			bookElement.setPublisher(publisher.item(i).getTextContent());
			bookElement.setCompany(company.item(i).getTextContent());
			bookElement.setAuthor(author.item(i).getTextContent());
			bookElement.setIsbn(ISBN.item(i).getTextContent());
			bookElement.setPrice(new Double(price.item(i).getTextContent()));
			try {
				bookElement.setUrl(new URL(url.item(i).getTextContent()));
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (DOMException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			bookList.add(bookElement);
		}
		return bookList;
	}

	public static void main(String[] arg) {

		String pathname = "xmldemo/books.xml";
		BookElementDOMParsing elementSAXParsing = new BookElementDOMParsing();
		Document document = null;
		document = elementSAXParsing.parseReadFile(pathname);
		List<BookElement> bookElements = elementSAXParsing.getBook(document
				.getDocumentElement());
		System.out.println("自定义JAVA类封装元素名称和内容");
		System.out.println(bookElements);
	}
}
